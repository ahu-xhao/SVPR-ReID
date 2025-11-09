# coding=utf-8
'''
@Time     : 2025/05/06 14:57:32
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import copy
from pathlib import Path
from matplotlib.pylab import cla
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
from .attention import Attention, CrossAttention, MLP, AttentionScoreMask, AttentionBlock, TwoWayTransformer,PromptRecapBlock
from .backbone.vit_view_decouple import vit_base_patch16_224_VDT
import logging

logger = logging.getLogger("CLIP-ReID.model")

factory = {
    'vit_base_patch16_224_VDT': vit_base_patch16_224_VDT,
}


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


class Header(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(Header, self).__init__()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        feat_bn = self.bottleneck(x)
        cls_score = self.classifier(feat_bn)

        return cls_score, feat_bn

class SeCap(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.pretrain_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.num_classes = num_classes
        self.cam_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        self.view_num = view_num if cfg.MODEL.SIE_VIEW else 0
        self.sie_coe = cfg.MODEL.SIE_COE

        # VDT init
        depth = 'base'
        num_depth = {'small': 8, 'base': 12, }[depth]
        num_heads = {'small': 8, 'base': 12, }[depth]
        mlp_ratio = {'small': 3., 'base': 4, }[depth]
        qkv_bias = {'small': False, 'base': True}[depth]
        qk_scale = {'small': 768 ** -0.5, 'base': None, }[depth]
        
        self.BACKBONE = factory[cfg.MODEL.NAME](img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE, sie_xishu=cfg.MODEL.SIE_COE,
                                                num_classes=num_classes, camera=self.cam_num, view=self.view_num,
                                                drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                # num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                inner_sub=True, local_feat=True)
        self.BACKBONE.load_param(self.pretrain_path)
        
        self.header = Header(self.in_planes, self.num_classes)
        self.header_view = Header(self.in_planes, view_num)
        self.header_out = Header(self.in_planes, self.num_classes)

        # local feature block (last 2 layer of VDT)
        block = self.BACKBONE.blocks[-1]
        layer_norm = self.BACKBONE.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        
        self.use_prm = True
        self.prompt_len=64
        self.prompt_trans_depth = 2
        # PRM init
        self.__init_query__(self.prompt_len, self.in_planes)
        self.prm = PromptRecapBlock(embedding_dim=self.in_planes, num_heads=num_heads)

        # LFRM init
        self.lfrm = TwoWayTransformer(
            # 层数
            depth=self.prompt_trans_depth,
            # 输入channel
            embedding_dim=self.in_planes,
            # MLP内部channel
            mlp_ratio=mlp_ratio,
            # attention的head数
            num_heads=num_heads,
            use_to_way=True,
        )

        self.image_pe = nn.Parameter(torch.zeros(1, self.BACKBONE.num_patches, self.in_planes))
        trunc_normal_(self.image_pe, std=.02)
        self.out_token = nn.Parameter(torch.zeros(1, self.in_planes))
        trunc_normal_(self.out_token, std=.02)

    def __init_query__(self, prompt_len, prompt_dim):
        self.prompt = nn.Parameter(torch.zeros(1, prompt_len, prompt_dim))  # type: ignore
        trunc_normal_(self.prompt, std=.02)
        
        
    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None,test_score=False):
        B = x.shape[0]
        # VDT
        local_features = self.BACKBONE(x, cam_label=cam_label)
        local_feat = self.b1(local_features)
        global_features = local_feat[:, 0:1]
        view_features = local_feat[:, 1:2]
        local_feat = local_features[:, 2:]
        inv_features = global_features - view_features

        # PRM
        query_feat = torch.repeat_interleave(self.prompt, B, dim=0)
        if self.use_prm:
            Re_Prompt = self.prm(query_feat, inv_features)
        else:
            Re_Prompt = query_feat

        # LFRM
        out_token = self.out_token.unsqueeze(0).expand(B, -1, -1)
        prompt = torch.cat((out_token, Re_Prompt), dim=1)
        pos_src = torch.repeat_interleave(self.image_pe, B, dim=0)
        prompts, img_feature = self.lfrm(local_feat, pos_src, prompt, cam_label)
        out_feat = prompts[:, 0:1, :]
        
        global_features = global_features.view(B,-1)
        view_features = view_features.view(B,-1)
        out_feat = out_feat.view(B,-1)
        cls_score, global_feat_bn = self.header(global_features)
        cls_score_out, out_feat_bn = self.header_out(out_feat)
        cls_score_view, view_feat_bn  = self.header_view(view_features)
        
        if self.training:
            
            return [cls_score, cls_score_out, cls_score_view], [global_features, out_feat, view_features]
        else:
            if test_score:
                return [cls_score,cls_score_out, cls_score_view], torch.cat([global_features, out_feat], dim=1)
            else:
                return torch.cat([global_features, out_feat], dim=1)
        

    def load_param_text(self, trained_path):
        # param_dict = torch.load(trained_path)
        # model_dict = self.state_dict()
        loaded_keys = []
        # for k, v in param_dict.items():
        #     new_k = k.replace('module.', '')
        #     if new_k.startswith('text_encoder') or new_k.startswith('prompt_learner'):
        #         if new_k in model_dict and model_dict[new_k].shape == v.shape:
        #             self.state_dict()[new_k].copy_(v)
        #             loaded_keys.append(new_k)
        print(f'Loaded {len(loaded_keys)} text-related parameters from {trained_path}')

    def load_param(self, trained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        param_dict = torch.load(trained_path, map_location=device)
        # param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
