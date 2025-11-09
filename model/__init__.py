# coding=utf-8
'''
@Time     : 2025/06/06 18:07:01
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib
import logging
logger = logging.getLogger(__name__)

from .make_model_clipreid_xhao import CLIP_ReID, CLIP_prompt, CLIP_PTS, CLIP_GCA, CLIP_AUG
from .make_model_clipreid_ours_v1 import CLIP_baseline, CLIP_inverseText_localPatchs_CVPR
# from .make_model_clipreid_ours_v2 import CLIP_text_prompt_localPatchs_CVPR,CLIP_text
from .make_model_clipreid_ours_v2 import *
from .make_model_vit import ViT, TransReID, VDT
from .make_model_vit_ours import TransReID_CVPR, ViT_CVPR, VDT_CVPR
from .make_model_vit_secap import SeCap


def make_model(cfg, num_class, camera_num, view_num, **kwargs):
    arch_name = cfg.MODEL.ARCH_NAME
    try:
        model = eval(arch_name)(num_class, camera_num, view_num, cfg, **kwargs)
    except NameError:
        logger.error(f"Model Name {arch_name} not found. Please check the model name.")
        raise
    logger.info(f"Model {arch_name} architecture:\n{model}")
    logger.info(f"Model {arch_name} created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    return model
