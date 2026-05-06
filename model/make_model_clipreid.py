# coding=utf-8
'''
@Time     : 2025/05/06 14:57:32
@Author   : XHao
@Email    : xhao2510@foxmail.com
'''
# here put the import lib

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import logging


logger = logging.getLogger("SVPR-ReID.model")


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
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
            B, N, C = image_features.shape

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        cls_score = self.classifier(feat)
        cls_score_proj = self.classifier_proj(feat_proj)
        if self.training:
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            test_feature = torch.cat([img_feature, img_feature_proj], dim=1)
            test_scores = [cls_score, cls_score_proj]
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


class CLIP_baseline(nn.Module):
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
        self.cam_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        self.view_num = view_num if cfg.MODEL.SIE_VIEW else 0
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

        self._make_cv_embed(self.cam_num, self.view_num)

    def _make_cv_embed(self, camera_num, view_num):
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

    def freeze_layers(self, moudle):
        for param in moudle.parameters():
            param.requires_grad = False

    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None, test_score=False):
        if self.cam_num > 1 and self.view_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif self.view_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        B = x.shape[0]
        image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
        B, N, C = image_features.shape

        img_feature_last = image_features_last[:, 0]
        img_feature = image_features[:, 0]
        # img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        # feat_proj = self.bottleneck_proj(img_feature_proj)

        cls_score = self.classifier(feat)
        # cls_score_proj = self.classifier_proj(feat_proj)
        if self.training:
            # return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
            return [cls_score], [img_feature]

        else:
            if test_score:
                return [cls_score], img_feature
            else:
                return img_feature

    def load_param(self, trained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param_dict = torch.load(trained_path, map_location=device)
        # param_dict = torch.load(trained_path)
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except Exception as e:
                print(
                    f'param_dict key {i}: {param_dict[i].shape} not match in model key {i.replace("module.", "")}: {self.state_dict()[i.replace("module.", "")].shape}')
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_text(self, trained_path):
        loaded_keys = []
        # param_dict = torch.load(trained_path)
        # model_dict = self.state_dict()
        # for k, v in param_dict.items():
        #     new_k = k.replace('module.', '')
        #     if new_k.startswith('text_encoder') or new_k.startswith('prompt_learner'):
        #         if new_k in model_dict and model_dict[new_k].shape == v.shape:
        #             self.state_dict()[new_k].copy_(v)
        #             loaded_keys.append(new_k)

        print(f'Loaded {len(loaded_keys)} text-related parameters from {trained_path}')


from .backbone.clip import clip


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size, use_view_token: bool = False, learnable_tokens: int = 1):
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = backbone_name

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(
        cfg,
        state_dict or model.state_dict(),
        h_resolution, w_resolution, vision_stride_size,
        use_view_token, learnable_tokens   # xhao add
    )

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

    def forward(self, label, view_label=None, time_label=None):
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

    def forward_(self, prompt_embeddings, eot_indices=None):
        """
        prompt_embeddings: shape (B, L, D), already contains inserted context vectors
        """
        x = prompt_embeddings + self.positional_embedding[:prompt_embeddings.shape[1], :].to(prompt_embeddings.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 默认取第一个 token（或根据任务取其他位置）
        if eot_indices is not None:
            x = x[torch.arange(x.shape[0]), eot_indices]  # shape (B, D)
        else:
            x = x[:, 0, :] @ self.text_projection  # shape (B, D)
        return x
