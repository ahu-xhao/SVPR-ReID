# coding=utf-8
'''
@Time     : 2025/05/06 14:57:32
@Author   : XHao
@Email    :  xhao2510@foxmail.com
'''
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from .attention import Attention, CrossAttention, MLP, AttentionScoreMask, CrossAttentionBlock
from .moe import MOE_simple, Router as AttentionRouter
from loss.softmax_loss import CrossEntropyLabelSmooth
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


class ContextVisionBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.cross_attn2 = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim)
        self.norm4 = norm_layer(dim)

    def forward(self, q, k):
        # self attention
        queries = q + self.attn(q)
        queries = self.norm1(queries)
        # cross attention
        queries = queries + self.cross_attn(queries, k)
        queries = self.norm2(queries)
        # mlp
        queries = queries + self.mlp(queries)
        queries = self.norm4(queries)

        # cross attention 2
        cross_out = self.cross_attn2(k, queries)
        k = k + cross_out
        k = self.norm3(k)

        return queries, k


class ContextVisionProgressiveRenfineNet(nn.Module):
    def __init__(self, text_dim=512, visual_dim=768, num_heads=8, num_layers=3):
        super().__init__()
        self.in_planes = visual_dim
        self.text_dim = text_dim
        self.topk = 0.5
        self.monument = 0.1
        self.vt_fuse = CrossAttention(dim=visual_dim, num_heads=num_heads)  # 融合视觉类别和文本特征
        shared_block = ContextVisionBlock(dim=visual_dim, num_heads=num_heads)
        self.blocks = nn.ModuleList(
            [shared_block for _ in range(num_layers)]
        )
        self.ln_norm = nn.LayerNorm(visual_dim)
        self.attn_masker = AttentionScoreMask(dim=self.in_planes, num_heads=num_heads)
        self.gate_fc = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_all(self, visual_cls, visual_patches, text_inverse):
        if visual_cls.dim() == 2:
            visual_cls = visual_cls.unsqueeze(1)
        if text_inverse.dim() == 2:
            text_inverse = text_inverse.unsqueeze(1)
           # 融合视觉类别和文本特征
        query = self.vt_fuse(torch.cat([visual_cls, text_inverse], dim=-1))  # (B, 1, C)
        key = visual_patches
        B, N, C = visual_patches.shape
        topk_indices_list = []

        for blk in self.blocks:
            # 计算注意力掩码和topk索引
            mask, topk_indices = self.attn_masker(query, key, top_k=self.topk, method="mean")  # (B, N)
            topk_indices_list.append(topk_indices)
            mask_expanded = mask.unsqueeze(-1).expand_as(key)  # (B, N, C)
            selected_key = key[mask_expanded].view(B, -1, C)  # (B, topk, C)
            query, updated_selected_key = blk(q=query, k=selected_key)
            # 更新原始visual_patches中的topk patches
            for batch_idx in range(B):
                indices = topk_indices[batch_idx]

                visual_patches[batch_idx, indices] = updated_selected_key[batch_idx].to(visual_patches.dtype)

            key = visual_patches

            visual_cls = self.monument * query + visual_cls
            text_inverse = self.monument * query + text_inverse

        return visual_cls, visual_patches, text_inverse

    def forward(self, visual_cls, visual_patches, attr_token=None, prompt=None):
        if visual_cls.dim() == 2:
            visual_cls = visual_cls.unsqueeze(1)
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(1)

        sematic = visual_cls + attr_token if attr_token is not None else visual_cls
        query = self.vt_fuse(prompt, sematic, sematic)  # (B,1,C)

        B, N, C = visual_patches.shape
        device = visual_patches.device

        current_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)  # (B,N)
        updated_patches = visual_patches.clone()

        all_selected_indices = []

        for blk in self.blocks:
            # 当前候选patch
            key = torch.gather(visual_patches, dim=1, index=current_indices.unsqueeze(-1).expand(-1, -1, C))  # (B,num_candidates,C)

            # 选 topk
            mask, topk_indices_in_current = self.attn_masker(query, key, top_k=self.topk, method="mean")  # mask (B,num_candidates)
            topk_indices_in_current = topk_indices_in_current.squeeze(1)  # (B,topk)
            # 当前阶段选中的在全局的索引
            topk_indices_global = torch.gather(current_indices, dim=1, index=topk_indices_in_current)  # (B,topk)
            all_selected_indices.append(topk_indices_global)

            # 根据mask选中tokens
            mask_exp = mask.unsqueeze(-1).expand_as(key)  # (B,num_candidates,C)
            selected_key = key[mask_exp].view(B, -1, C)  # (B,topk,C)
            query, updated_selected = blk(q=query, k=selected_key)

            updated_patches.scatter_(1, topk_indices_global.unsqueeze(-1).expand(-1, -1, C), updated_selected.to(updated_patches.dtype))

            current_indices = topk_indices_global

            # 融合query
            gate = self.gate_fc(query)  # (B,C)
            visual_cls = gate * query + (1 - gate) * visual_cls
            if prompt is not None:
                prompt = gate * query + (1 - gate) * prompt
            # visual_cls = self.monument * query + visual_cls
            # if text_inverse is not None:
            #     text_inverse = self.monument * query + text_inverse
        return visual_cls, updated_patches, prompt


class ViewPromptBuilder(nn.Module):
    def __init__(self, cfg, dtype, token_embedding, text_length=None):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.token_embedding = token_embedding
        self.n_prompt_view = cfg.MODEL.VIEW_PROMPT
        ctx_template = f"A photo of a {'X ' * self.n_prompt_view} person from X view."

        # 预处理模板，避免重复计算
        # self.tokenizer = SimpleTokenizer()
        self.text_length = text_length if text_length is not None else cfg.MODEL.TEXT_LEN
        self.truncate = True
        self.tokenized_prompts = clip.tokenize(ctx_template, context_length=self.text_length, truncate=self.truncate).cuda()
        with torch.no_grad():
            token_prefix_suffix = token_embedding(self.tokenized_prompts).type(dtype).cuda()
        self.register_buffer("token_prefix_suffix", token_prefix_suffix)

        self.tokenized_view = clip.tokenize("ground aerial", context_length=self.text_length, truncate=self.truncate).cuda()
        with torch.no_grad():
            token_view = token_embedding(self.tokenized_view).type(dtype).cuda()
        self.register_buffer("token_view", token_view)
        # 记录 "X" 对应的 token ID（仅用于查找占位符）
        self.x_token_id = clip.tokenize(["X"], context_length=self.text_length, truncate=self.truncate).cuda()[0][1]

    def forward(self, view_label, prompts=None):
        b = view_label.shape[0]
        prompts = prompts.expand(b, -1, -1)
        token_prefix_suffix = self.token_prefix_suffix.expand(b, -1, -1).clone()

        # 使用固定的 token_view
        token_view = self.token_view.expand(b, -1, -1).clone()
        token_view = token_view[:, 1:, :]
        view_label_token = token_view[torch.arange(b, device=view_label.device), view_label]  # 获取每个视图标签对应的 token
        view_label_token = view_label_token
        x_pos = (self.tokenized_prompts == self.x_token_id).nonzero(as_tuple=True)[1]
        for j in range(self.n_prompt_view):  # 从第1个 "X" 开始填充
            token_idx = x_pos[j].item()
            token_prefix_suffix[:, token_idx, :] = prompts[:, j, :]
        # 用 aerial or ground 填充视角 "X" 的位置
        token_prefix_suffix[:, x_pos[-1]] = view_label_token

        return token_prefix_suffix


class HybirdTextEncoder(nn.Module):
    def __init__(self, cfg, dtype, token_embedding):
        super().__init__()
        self.use_text = cfg.MODEL.USE_TEXT
        self.text_format = cfg.MODEL.TEXT_FORMAT
        self.use_attr = cfg.MODEL.USE_ATTR
        self.num_text_prompt = cfg.MODEL.TEXT_PROMPT
        self.num_view_prompt = cfg.MODEL.VIEW_PROMPT
        self.model_name = cfg.MODEL.NAME
        self.in_planes_proj = 512 if cfg.MODEL.NAME == 'ViT-B-16' else 1024
        self.text_length = cfg.MODEL.TEXT_LEN

        scale = self.in_planes_proj ** -0.5
        self.text_prompt = nn.Parameter(scale * torch.randn(1, self.num_text_prompt, self.in_planes_proj)) if self.num_text_prompt > 0 else None
        self.view_prompt = nn.Parameter(scale * torch.randn(1, self.num_view_prompt, self.in_planes_proj)) if self.num_view_prompt > 0 else None
        self.view_promptlearner = ViewPromptBuilder(cfg, dtype, token_embedding, text_length=self.text_length)
        self.CA_text2prompt = CrossAttention(dim=self.in_planes_proj)
        self.view_fusion = nn.Linear(self.in_planes_proj * self.num_view_prompt, self.in_planes_proj) if self.num_view_prompt > 0 else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_one(self, clip_model, text_tokens, view_label):
        # use_text = self.use_text and text_tokens is not None
        use_text = text_tokens is not None
        if use_text:
            text_features_raw = clip_model.encode_text(text_tokens, text_prompt=self.text_prompt)
            text_feat_global = text_features_raw[torch.arange(text_features_raw.shape[0]), text_tokens.argmax(dim=-1)]
            if self.num_text_prompt > 0:
                text_prompt = text_features_raw[:, 5:5 + self.num_text_prompt]
                text_prompt = torch.mean(torch.cat([text_prompt, text_feat_global.unsqueeze(1)], dim=1), dim=1)  # [B, C]
                view_prompt = text_features_raw[:, 7 + self.num_text_prompt]  # [B, C]
                # text2prompt = self.CA_text2prompt(text_prompt, text_feat_global.unsqueeze(1))# [B, 1, C]
                # text_prompt = text_prompt + text2prompt.squeeze(1)  # [B, C]
            else:
                text_prompt = None  # [B, C]
                view_prompt = None
        else:
            text_prompt = torch.mean(self.text_prompt.expand(text_tokens.shape[0], -1, -1), dim=1) if self.num_text_prompt > 0 else None
            text_feat_global = None  # [B, C]
            view_prompt = None
        return text_prompt, text_feat_global, view_prompt

    def forward(self, clip_model, text_tokens=None, view_label=None):

        # 提取文本描述特征
        text_token = None
        use_text = self.use_text and text_tokens is not None
        if use_text:
            text_features = clip_model.encode_text(text_tokens, text_prompt=self.text_prompt)
            text_cls = text_features[torch.arange(text_features.shape[0]), text_tokens.argmax(dim=-1)]
            text_prompts = text_features[:, 5:5 + self.num_text_prompt]
            text_token = text_cls + torch.mean(text_prompts, dim=1)  # [B, C]

        # 提取视角描述特征
        view_token = None
        if self.num_view_prompt > 0:
            view_prompt_embedding = self.view_promptlearner(view_label, self.view_prompt)
            view_prompt_features = clip_model.encode_text(view_prompt_embedding)
            # 提取文本提示
            view_prompts = view_prompt_features[:, 5:5 + self.num_view_prompt]
            # view_prompt = torch.mean(view_prompt, dim=1)

            view_prompt_word = view_prompt_features[:, 7 + self.num_view_prompt:8 + self.num_view_prompt]  # [B, 1, C]
            view_word2prompt = self.CA_text2prompt(q=view_prompts, k=view_prompt_word, v=view_prompt_word)  # [B, num_view_prompt ,C]
            # view_token = torch.mean(view_token, dim=1)  # [B, C]
            view_fuse = view_word2prompt + view_prompts  # [B, num_view_prompt ,C]
            view_fuse = view_fuse.view(view_fuse.shape[0], -1)  # [B, C * num_view_prompt]
            view_token = self.view_fusion(view_fuse)  # [B, C]

        return text_token, view_token


class AttributeScatterMOE(nn.Module):
    def __init__(self, cfg, embed_dim=768):
        super().__init__()
        root = cfg.DATASETS.ROOT_DIR
        attr_names = cfg.DATASETS.ATTR_NAME
        attr_classes = cfg.DATASETS.ATTR_CLASS
        self.attribute_num_classes = {k: v for k, v in zip(attr_names, attr_classes)}
        self.attr_list = attr_names
        self.num_attrs = len(self.attr_list)

        self.attributesPrompt = nn.Parameter(torch.randn(1, self.num_attrs, embed_dim)) if self.num_attrs > 0 else None
        self.cross_attention = CrossAttention(dim=embed_dim, num_heads=8)
        self.attributesMOE = MOE_simple(input_dim=embed_dim, num_experts=4, topk=3)
        self.attributesRouter = MLP(embed_dim, 1)
        self.attributsHeader = nn.ModuleDict()
        for attr, num_classes in self.attribute_num_classes.items():
            self.attributsHeader[attr] = Header(embed_dim=embed_dim, num_classes=num_classes)
        self.use_smooth = cfg.MODEL.IF_LABELSMOOTH == 'on'
        self.xent = F.cross_entropy
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text_cls, visual_cls, visual_patchs, attr_labels=None, topk_ratio=0.7):

        B, N, C = visual_patchs.shape
        attr_expert_outputs = []
        attr_logits = {}
        attr_scores = []
        # text_cls = text_cls.unsqueeze(1)  # [B, 1, C]
        # visual_cls = visual_cls.unsqueeze(1)  # [B, 1, C]
        # 1. 遍历每个属性，提取 expert 输出、分类 logits、计算与文本相似度
        attribute_prompt = self.attributesPrompt.expand(B, -1, -1) if self.attributesPrompt is not None else None
        moe_in = self.cross_attention(q=text_cls, k=visual_patchs, v=visual_patchs)  # [B, 1, C]
        moe_in = moe_in.squeeze(1)  # [B, C]

        for i, attr in enumerate(self.attr_list):
            header = self.attributsHeader[attr]
            attr_in = moe_in + attribute_prompt[:, i, :]  # [B, C]
            attr_in = attr_in + visual_cls
            moe_out = self.attributesMOE(attr_in)
            moe_out = moe_out.squeeze(1)  # [B, C]
            # moe_out = moe_out + visual_cls  # [B, C]
            attr_expert_outputs.append(moe_out)
            logits, expert_out_bn = header(moe_out)  # [B, num_classes]
            attr_logits[attr] = logits

            # 属性打分
            score = self.attributesRouter(attr_in)  # [B, C]
            score = score.squeeze(1)  # [B, C]
            score_mean = score.mean(dim=-1)  # [B]
            attr_scores.append(score_mean)

        # 2. 堆叠专家输出和相似度得分
        attr_expert_outputs = torch.stack(attr_expert_outputs, dim=1)  # [B, A, C]
        attr_scores = torch.stack(attr_scores, dim=1)  # [B, A]

        # 3. Top-k 属性选择用于增强
        top_k = int(self.num_attrs * topk_ratio)
        _, topk_indices = torch.topk(attr_scores, top_k, dim=1)  # [B, k]

        # gather expert outputs and scores
        topk_expert_outputs = attr_expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, k, C]
        topk_scores = attr_scores.gather(1, topk_indices)  # [B, k]

        # 加权组合（加一个 softmax 稳定）
        weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)  # [B, k, 1]
        enhanced_feat = (topk_expert_outputs * weights).sum(dim=1)  # [B, C]
        # 4. 分类损失计算（所有属性）
        loss_attr = 0.
        if attr_labels is not None:
            for i, attr in enumerate(self.attr_list):
                logits = attr_logits[attr]  # [B, num_classes]
                labels = attr_labels[:, i]  # [B]
                loss_attr += self.xent(logits, labels)

        return enhanced_feat, loss_attr


class Header(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        feat_bn = self.bottleneck(x)
        cls_score = self.classifier(feat_bn)
        return cls_score, feat_bn


class CLIP_SVPR_ReID(nn.Module):
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

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg, self.pretrain_path, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.BACKBONE = clip_model

        self._make_cv_embed(self.cam_num, self.view_num)
        # text: captions or attributes component
        self.use_text = cfg.MODEL.USE_TEXT
        if self.use_text:
            self.num_text_prompt = cfg.MODEL.TEXT_PROMPT
            self.TextNet = HybirdTextEncoder(cfg, clip_model.dtype, clip_model.token_embedding)
            self.inverseNet_T2I = MLP(in_dim=self.in_planes_proj, hidden_dim=self.in_planes_proj * 4, output_dim=self.in_planes, drop=0.1)
            self.inverseNet_P2I = MLP(in_dim=self.in_planes_proj, hidden_dim=self.in_planes_proj * 4, output_dim=self.in_planes, drop=0.1)
            # self.header_text = Header(embed_dim=self.in_planes_proj, num_classes=self.num_classes)
        logger.info(f"use text : {self.use_text}")

        self.n_view_prompt = cfg.MODEL.VIEW_PROMPT
        self.use_view = self.n_view_prompt > 0
        if self.use_view:
            self.VBD = ViewBidirectionalDecoupling_block(dim=self.in_planes, use_alpha=True)
            self.header_view = Header(embed_dim=self.in_planes, num_classes=3)

        logger.info(f"use view : {self.use_view}")

        # attributes moe
        self.use_attr = cfg.MODEL.USE_ATTR
        if self.use_attr:
            self.ASMOE = AttributeScatterMOE(cfg, embed_dim=self.in_planes)
        logger.info(f"use attr : {self.use_attr}")

        self.use_cvpr = True
        if self.use_cvpr:
            self.use_cvpr_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))  # CVPR token for text
            trunc_normal_(self.use_cvpr_token, std=.02)
            self.CVPRNet = ContextVisionProgressiveRenfineNet(text_dim=self.in_planes_proj, visual_dim=self.in_planes)
            self.header_cvpr = Header(embed_dim=self.in_planes, num_classes=self.num_classes)
            # logger.info(f"use CVPRNet model is {self.CVPRNet}")
        logger.info(f"use cvpr : {self.use_cvpr}")

        # local feature component
        self.use_visual_local_feat = True
        if self.use_visual_local_feat:
            # self.num_patches = self.h_resolution * self.w_resolution
            self.num_patches = 128
            self.local_weight = nn.Parameter(torch.randn(self.num_patches, self.in_planes))
            self.header_local = Header(embed_dim=self.in_planes, num_classes=self.num_classes)
        logger.info(f"use visual local feat : {self.use_visual_local_feat}")

        self.header = Header(embed_dim=self.in_planes, num_classes=self.num_classes)
        # self.header_text = Header(embed_dim=self.in_planes, num_classes=self.num_classes)

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

    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None, test_score=False):
        if self.cam_num > 1 and self.view_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif self.view_num > 1:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None

        # 处理文本特征
        if text is not None and text[0] is not None:
            text_tokenizeds, attr_labels = text[0], text[1]
            text_prompt_raw, view_prompt_raw = self.TextNet(self.BACKBONE, text_tokenizeds, view_label)  # B,512
            text_prompt = self.inverseNet_T2I(text_prompt_raw)  # B,768
            view_prompt = self.inverseNet_P2I(view_prompt_raw)
        else:
            text_tokenizeds, attr_labels = None, text[1] if text is not None else None
            text_prompt, view_prompt = None, None

        # 处理图像特征,文本一引导图像视角感知（cat进clip visual 最后一层）
        image_features_last, image_features, image_features_proj = self.BACKBONE.encode_image(x, cv_embed, prompt=view_prompt.unsqueeze(1))
        B, N, C = image_features.shape

        img_cls = image_features[:, 0]
        img_patchs = image_features[:, 1:-1] if view_prompt is not None else image_features[:, 1:]
        view_prompt = image_features[:, -1] if view_prompt is not None else None

        # view decouple
        img_cls_inv, view_prompt = self.VBD(img_cls, view_prompt)  # [B, C], [B, C]

        if self.use_attr and attr_labels is not None:
            attr_feat_img, loss_attr = self.ASMOE(text_prompt, img_cls_inv, img_patchs, attr_labels)  # [B,C]
        else:
            attr_feat_img, loss_attr = None, torch.tensor(0.0, device=x.device)

        if self.use_cvpr:
            CVPR_query = self.use_cvpr_token.expand(B, -1, -1)  # [B, 1, C]
            CVPR_v_cls, CVPR_v_patches, CVPR_query = self.CVPRNet(img_cls_inv, img_patchs, attr_feat_img, CVPR_query)
            # residual connection
            img_patchs = img_patchs + CVPR_v_patches.view(B, -1, C) * 0.1
            CVPR_query = CVPR_query.view(B, C)  # [B, C]

        if self.use_visual_local_feat:
            local_weight = self.local_weight.repeat(B, 1, 1)
            local_weight = torch.tanh(local_weight)  # 对 local_weight 应用 tanh 激活函数
            img_patch = torch.mean(img_patchs * local_weight, dim=1)

        cls_score, feat = self.header(img_cls_inv)
        return_score = [cls_score]
        return_feats = [img_cls_inv]

        if self.use_cvpr:
            cls_score_cvpr, feat_cvpr = self.header_cvpr(CVPR_query)
            return_score.append(cls_score_cvpr)
            return_feats.append(CVPR_query)

        if self.use_visual_local_feat:
            cls_score_patch, feat_patch = self.header_local(img_patch)
            return_score.append(cls_score_patch)
            return_feats.append(img_patch)

        if self.use_attr:
            return_score.append(loss_attr)
            return_feats.append(attr_feat_img)

        return_feats.append(image_features_last[:, 0])  # [B, C] for image features

        cls_score_view, feat_view = self.header_view(view_prompt)  # [B, C]
        return_score.append(cls_score_view)  # [B, C] for image features
        return_feats.append(view_prompt)  # [B, C] for image features

        if self.training:
            return return_score, return_feats
        else:
            if test_score:
                # return return_score, torch.cat([*return_feats[:-3], return_feats[-2]], dim=1)
                return return_score, torch.cat([img_cls_inv, CVPR_query, img_patch, attr_feat_img], dim=1)
            else:
                # return torch.cat([*return_feats[:-3], return_feats[-2]], dim=1)
                return torch.cat([img_cls_inv, CVPR_query, img_patch, attr_feat_img], dim=1)

    def get_model_param_size(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params / 1e6  # in millions (M)

    def load_param_text(self, trained_path):
        loaded_keys = []
        print(f'Loaded {len(loaded_keys)} text-related parameters from {trained_path}')

    def load_param(self, trained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param_dict = torch.load(trained_path, map_location=device)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


class ViewBidirectionalDecoupling_block(nn.Module):
    def __init__(self, dim, use_alpha=True):
        super().__init__()
        self.use_alpha = use_alpha

        self.CA_view2cls = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.CA_cls2view = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0))  # 控制解耦程度

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, img_cls, view_prompt):
        # img_cls: [B, C] → [B, 1, C]
        img_cls = img_cls.unsqueeze(1)  # [B, C] → [B, 1, C]
        view_prompt = view_prompt.unsqueeze(1)  # [B, C] → [B, 1, C]

        # View-to-Cls Cross Attention
        view_to_cls, _ = self.CA_view2cls(query=img_cls, key=view_prompt, value=view_prompt)
        global_feat_inv = img_cls - self.alpha * view_to_cls if self.use_alpha else img_cls - view_to_cls
        global_feat_inv = global_feat_inv.squeeze(1)  # [B, C]

        # Cls-to-View Cross Attention
        cls_to_view, _ = self.CA_cls2view(query=view_prompt, key=global_feat_inv.unsqueeze(1), value=global_feat_inv.unsqueeze(1))
        view_prompt = view_prompt - cls_to_view  # 解耦后的 view prompt
        view_prompt = self.norm(view_prompt)
        view_prompt = self.mlp(view_prompt)
        view_prompt = view_prompt.mean(dim=1)  # [B, C]

        return global_feat_inv, view_prompt


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
