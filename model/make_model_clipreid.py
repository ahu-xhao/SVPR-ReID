# coding=utf-8
'''
@Time     : 2025/05/08 00:38:13
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import torch
import torch.nn as nn
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import test
from .attention import MLP, CrossAttention, Attention, AttentionScoreMask


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class SematicPromptBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward_self_cross(self, q, k, v=None):
        _, N_q, _ = q.shape
        # self attention
        q_self = torch.cat((q, k), dim=1)
        self_out = self.attn(q_self)
        queries = self_out[:, :N_q, :]
        queries = q + queries
        # cross attention
        k_cross = self_out[:, N_q:, :]
        q_cross = self.norm1(queries)
        cross_out = self.cross_attn(q_cross, k_cross)
        queries = queries + cross_out

        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries

    def forward_cross_self(self, q, k, v):
        attn_out = self.cross_attn(q, k)
        queries = q + attn_out
        queries = self.norm1(queries)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries


class SematicTransformer(nn.Module):
    def __init__(self, dim, num_classes, text_dim=512, num_heads=8):
        super().__init__()
        self.in_planes = dim
        self.num_classes = num_classes
        # self.sematic_prompt = nn.Parameter(torch.zeros(1, self.in_planes))
        # trunc_normal_(self.sematic_prompt, std=.02)
        self.sematic_block1 = SematicPromptBlock(dim=text_dim, num_heads=8)
        self.sematic_block2 = SematicPromptBlock(dim=self.in_planes, num_heads=8)
        self.sematic_block3 = SematicPromptBlock(dim=self.in_planes, num_heads=8)

        self.prompt2text = nn.Linear(self.in_planes, text_dim, bias=False)
        self.text2prompt = nn.Linear(text_dim, self.in_planes, bias=False)
        self.attn_masker = AttentionScoreMask(dim=self.in_planes, num_heads=num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, sematic_prompt, text, cls_token, img_feats, topk=0.5):
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        B, N, C = img_feats.shape
        # topk = int(topk * N)

        # prompt learn
        sematic_textdim = self.prompt2text(sematic_prompt)
        sematic_textdim = self.sematic_block1.forward_cross_self(sematic_textdim, text, text)
        sematic_promptdim = self.text2prompt(sematic_textdim)

        cls_token = cls_token.unsqueeze(1)
        sematic_prompt = self.sematic_block2.forward_cross_self(sematic_promptdim, cls_token, cls_token)

        mask = self.attn_masker(cls_token, img_feats, top_k=topk, method="mean")

        # sematic_tokens = [img_feats[i][mask[i]] for i in range(B)]
        # sematic_tokens = img_feats[torch.arange(B).unsqueeze(-1), mask, :]
        # indices = [torch.nonzero(mask[b], as_tuple=True)[0] for b in range(B)]
        # sematic_tokens = img_feats[torch.arange(B).unsqueeze(-1), indices, :].view(B, -1, C)
        sematic_tokens = img_feats[mask.unsqueeze(-1).expand_as(img_feats)].view(B, -1, C)
        sematic_prompt = self.sematic_block3.forward_cross_self(sematic_prompt, sematic_tokens, sematic_tokens)
        return sematic_prompt.squeeze(1)


class CLIP_ReID(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.pretrain_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view_num = view_num if cfg.MODEL.SIE_VIEW else 0
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg, self.pretrain_path, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if camera_num > 1 and view_num > 1:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif camera_num > 1:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif view_num > 1:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))
        self.cam_num = camera_num
        self.view_num = view_num
        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

        # self.sematic_prompt = nn.Parameter(torch.zeros(1, self.in_planes))
        # trunc_normal_(self.sematic_prompt, std=.02)
        # self.SeT = SematicTransformer(dim=self.in_planes, num_classes=num_classes, text_dim=self.in_planes_proj, num_heads=8)
        # self.classifier_sematic = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier_sematic.apply(weights_init_classifier)
        # self.bottleneck_sematic = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_sematic.bias.requires_grad_(False)
        # self.bottleneck_sematic.apply(weights_init_kaiming)

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None, time_label=None, text=None, test_score=False):

        if get_text == True:
            prompts = self.prompt_learner(label, view_label, time_label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if self.cam_num > 1 and self.view_num > 1:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif self.cam_num > 1:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif self.view_num > 1:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            # visual_prompt = self.visual_prompt.expand(B, -1, -1)
            # visual_prompt = visual_prompt.view(B, self.n_visual_prompt, -1)
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
            B, N, C = image_features.shape

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        # 语义token
        if text is not None:
            sematic_prompt = self.sematic_prompt.unsqueeze(0).expand(B, -1, -1)
            sematic_prompt = self.SeT(sematic_prompt, image_features_proj[:, 0:1], image_features[:, 0], image_features_last[:, 1:])
            feat_sematic = self.bottleneck_sematic(sematic_prompt)
        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)
        if text is not None:
            cls_score_sematic = self.classifier_sematic(feat_sematic)
        if self.training:
            if text is not None:
                return [cls_score, cls_score_proj, cls_score_sematic], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
            else:
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            test_feature = torch.cat([img_feature, img_feature_proj, sematic_prompt],
                                     dim=1) if text is not None else torch.cat([img_feature, img_feature_proj], dim=1)
            test_scores = [cls_score, cls_score_proj, cls_score_sematic] if text is not None else [cls_score, cls_score_proj]
            if test_score == True:
                return test_scores, test_feature
            else:
                return test_feature

    def load_param(self, trained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_text(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        model_dict = self.state_dict()
        loaded_keys = []
        for k, v in param_dict.items():
            new_k = k.replace('module.', '')
            if new_k.startswith('text_encoder') or new_k.startswith('prompt_learner'):
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    self.state_dict()[new_k].copy_(v)
                    loaded_keys.append(new_k)
        print(f'Loaded {len(loaded_keys)} text-related parameters from {trained_path}')

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


from .backbone.clip import clip


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = backbone_name

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(cfg, state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # ctx_init = "A photo of a X X X X person."
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, view_label, time_label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CVTPromptLearner(nn.Module):
    def __init__(self, num_classes, dataset_name, dtype, token_embedding, n_ctx=4):
        super().__init__()
        self.dtype = dtype
        self.token_embedding = token_embedding
        self.n_prompt = n_ctx
        self.ctx_dim = 512
        self.num_classes = num_classes
        self.num_views = 2
        self.num_times = 2

        # 选择 prompt 模板
        if dataset_name in ["VehicleID", "veri"]:
            ctx_template = "A photo of a {} vehicle from {} view in {} time."
        else:
            ctx_template = "A photo of a {} person from {} view in {} time."
        ctx_template = ctx_template.format("X " * self.n_prompt, "X", "X")

        tokenized_prompts = clip.tokenize(ctx_template).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        ctx_other = "ground aerial night day"
        tokenized_other = clip.tokenize(ctx_other).cuda()
        with torch.no_grad():
            embedding_other = token_embedding(tokenized_other).type(dtype)
        self.tokenized_other = tokenized_other  # torch.Tensor

        # 初始化可学习的 context embedding
        self.ctx_cls = nn.Parameter(torch.randn(num_classes, self.n_prompt, self.ctx_dim, dtype=dtype))
        self.ctx_view = nn.Parameter(torch.randn(self.num_views, self.n_prompt, self.ctx_dim, dtype=dtype))
        self.ctx_time = nn.Parameter(torch.randn(self.num_times, self.n_prompt, self.ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_cls, std=0.02)

        # 定义 view 和 time 的可读名称
        self.view_text = {0: "ground", 1: "aerial"}
        self.time_text = {0: "night", 1: "day"}

        self.register_buffer("token_prefix", embedding[:, :5, :])
        self.register_buffer("token_suffix1", embedding[:, 5 + self.n_prompt:7 + self.n_prompt, :])
        self.register_buffer("token_suffix2", embedding[:, 8 + self.n_prompt:10 + self.n_prompt, :])
        self.register_buffer("token_suffix3", embedding[:, 11 + self.n_prompt:, :])
        self.register_buffer("token_other", embedding_other[:, 1:5, :])

    def forward(self, label, view_label, time_label):
        cls_ctx = self.ctx_cls[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix1 = self.token_suffix1.expand(b, -1, -1)
        suffix2 = self.token_suffix2.expand(b, -1, -1)
        suffix3 = self.token_suffix3.expand(b, -1, -1)
        other = self.token_other.expand(b, -1, -1)
        view_label = view_label.unsqueeze(1)
        view_emb = other[torch.arange(b).unsqueeze(-1), view_label, :]
        time_label = time_label.unsqueeze(1)
        time_emb = other[torch.arange(b).unsqueeze(-1), self.num_views + time_label, :]

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix1,  # (n_cls, *, dim)
                view_emb,  # (n_cls, *, dim)
                suffix2,  # (n_cls, *, dim)
                time_emb,  # (n_cls, *, dim)
                suffix3,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class HybirdPromptLearner(nn.Module):
    def __init__(self, num_classes, dataset_name, dtype, token_embedding):
        super().__init__()
        self.dtype = dtype
        self.token_embedding = token_embedding
        self.n_ctx = 4
        self.n_cls_ctx = 4
        self.ctx_dim = 512
        self.num_classes = num_classes

        # 根据数据集选择模板
        if dataset_name in ["VehicleID", "veri"]:
            self.ctx_template = "A photo of a {} vehicle from {} view in {} time."
        else:
            self.ctx_template = "A photo of a {} person from {} view in {} time."

        # 初始化可学习的 context embeddings（共享的）
        # shape: (n_ctx, ctx_dim)
        ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # 占位符 token 对应的实际词（如 "X X X X"）
        # 为了计算总 token 长度，需要一个 dummy 模板
        # dummy_text = self.ctx_template.format("X" * 4, "ground", "day")
        # tokenized = clip.tokenize([dummy_text])
        # self.tokenized_template = tokenized  # 保存模板 token shape（用于后续生成 mask）

    def forward(self, label, view_label, time_label):
        view_text = {0: "ground", 1: "aerial"}
        time_text = {0: "night", 1: "day"}
        # 动态构造文本
        texts = [
            self.ctx_template.format("X X X X", view_text[view], time_text[time])
            for view, time in zip(view_label, time_label)
        ]
        tokenized = clip.tokenize(texts).cuda()  # (B, L)

        with torch.no_grad():
            embeddings = self.token_embedding(tokenized).type(self.dtype)  # (B, L, D)

        # 找到 "[X X X X]" 的位置，将其替换为可学习参数 self.ctx
        # 这里我们假设这 4 个 "X" 是连续的，在 tokenizer 中会变成 4 个 token
        # 找出 "X" 开头的位置
        x_token_id = clip.tokenize(["X"])[0][1]  # 找到 "X" 的 token id
        x_pos = (tokenized == x_token_id).nonzero(as_tuple=False).view(-1, self.n_ctx)

        # 替换对应位置的 embedding
        for i in range(embeddings.shape[0]):
            embeddings[i, x_pos[i]] = self.ctx

        return embeddings  # (B, L, D)
