# coding=utf-8
'''
@Time     : 2024/04/09 12:42:48
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib
from typing import Mapping
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def block_features(self):
        if not hasattr(self, 'backbone'):
            self.print('backbone is not defined', color='red')
            block_features = []
        elif len(self.backbone.block_features) == 0:
            self.print('block_features is empty', color='red')
            block_features = []
        else:
            block_features = self.backbone.block_features
        return block_features

    def load_param(self, pth_path):
        param_dict = torch.load(pth_path)
        self.load_dict(param_dict)
        print('Loading pretrained model from {}'.format(pth_path))

    def load_dict(self, param_dict):
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'norm_layer' in i:
                i_ = i.replace('norm_layer', 'layer_norm')
                self.state_dict()[i_].copy_(param_dict[i])
            else:
                try:
                    self.state_dict()[i].copy_(param_dict[i])
                except (KeyError):
                    self.print(f'KeyError: {i}', color='red')
                    continue

    def print(self, c, color='red', *args, **kwargs):
        start = {
            'red': '\033[0;31m',
            'green': '\033[0;32m',
            'yellow': '\033[0;33m',
            'blue': '\033[0;34m',
            'purple': '\033[0;35m',
            'cyan': '\033[0;36m',
            'white': '\033[0;37m',
        }
        end = '\033[0m'
        print(f"{start[color]}{c}{end}", args, kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        This method should be overridden by subclasses to define the specific forward behavior.
        """
        raise NotImplementedError("The forward method must be implemented by subclasses.")
