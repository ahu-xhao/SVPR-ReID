# coding=utf-8
'''
@Time     : 2025/05/06 14:57:32
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import copy
import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
import logging
from .backbone.vit_trans import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbone.vit_view_decouple import vit_base_patch16_224_VDT
logger = logging.getLogger("CLIP-ReID.model")
factory = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
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
   
    
class ViT(nn.Module):
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

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        # self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        # self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.BACKBONE = factory[cfg.MODEL.NAME](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                num_classes=num_classes,
                                                camera=self.cam_num, view=self.view_num,
                                                stride_size=cfg.MODEL.STRIDE_SIZE,
                                                drop_path_rate=cfg.MODEL.DROP_PATH,
                                                drop_rate=cfg.MODEL.DROP_OUT,
                                                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        self.BACKBONE.load_param(self.pretrain_path)

    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None, test_score=False):
        features = self.BACKBONE(x)
        global_feat = features[:, 0]
        B, N, C = features.shape

        global_feat_bn = self.bottleneck(global_feat)
        cls_score = self.classifier(global_feat_bn)

        if self.training:
            return [cls_score], [global_feat]
        else:
            if test_score:
                return [cls_score], global_feat
            else:
                return global_feat

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


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat(
        [features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


class TransReID(nn.Module):
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

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # Initialize the backbone model
        self.BACKBONE = factory[cfg.MODEL.NAME](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, stride_size=cfg.MODEL.STRIDE_SIZE,
                                                num_classes=num_classes, camera=self.cam_num, view=self.view_num,
                                                drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                local_feature=True)
        self.BACKBONE.load_param(self.pretrain_path)
        # Last block and layer norm for JPM of TransReID
        block = self.BACKBONE.blocks[-1]
        layer_norm = self.BACKBONE.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        # JPM Settings
        self.shuffle_groups = 2
        self.shift_num = 5
        self.divide_length = 4
        self.rearrange = True
        logger.info(
            f"using rearrange size:{self.rearrange}\nusing shuffle_groups size:{self.shuffle_groups}\nusing shift_num size:{self.shift_num}\nusing divide_length size:{self.divide_length}")

        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None, test_score=False):
        features = self.BACKBONE(x)
        B, N, C = features.shape

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat_bn = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        cls_score_1 = self.classifier_1(local_feat_1_bn)
        cls_score_2 = self.classifier_2(local_feat_2_bn)
        cls_score_3 = self.classifier_3(local_feat_3_bn)
        cls_score_4 = self.classifier_4(local_feat_4_bn)

        global_feat_bn = self.bottleneck(global_feat)
        cls_score = self.classifier(global_feat_bn)

        if self.training:
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]
        else:
            if test_score:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1),
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

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


class VDT(nn.Module):
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
        qk_scale = {'small': 768 ** -0.5, 'base': None}[depth]
        
        self.BACKBONE = factory[cfg.MODEL.NAME](img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE, sie_xishu=cfg.MODEL.SIE_COE,
                                                num_classes=num_classes, camera=self.cam_num, view=self.view_num,
                                                drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                # num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                inner_sub=True)
        self.BACKBONE.load_param(self.pretrain_path)
        
        self.header = Header(self.in_planes, self.num_classes)
        self.header_view = Header(self.in_planes, view_num)
        
    def forward(self, x=None, label=None, cam_label=None, view_label=None, time_label=None, text=None,test_score=False):
        B = x.shape[0]
        features = self.BACKBONE(x, cam_label=cam_label)
        global_features = features[:, 0:1]
        view_features = features[:, 1:2]
        # local_feat = features[:, 2:]
        # inv_features = global_features - view_features
        
        global_features = global_features.view(B,-1)
        view_features = view_features.view(B,-1)
       
        cls_score, global_feat_bn = self.header(global_features)
        cls_score_view, view_feat_bn  = self.header_view(view_features)
        if self.training:
            return [cls_score, cls_score_view], [global_feat_bn, view_feat_bn]
        elif test_score:
            return [cls_score, cls_score_view], torch.cat([global_features, view_features], dim=1)
        else:
            return torch.cat([global_features, view_features], dim=1)
