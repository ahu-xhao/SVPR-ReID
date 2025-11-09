# coding=utf-8
'''
@Time     : 2023/12/26 09:39:54
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib
import glob
import json
import os
import sys

import mat4py
import pandas as pd
import torch
sys.path.append(os.getcwd())
import os.path as osp
from .bases import BaseImageDataset
import re
import logging

__all__ = ['AGReIDv2', ]

ATTR_MAP = {
    'gender': ['male', 'female', 'unknown'],
    'age': ['young', 'middle', 'old', 'unknown'],
    'height': ['child', 'short', 'medium', 'tall', 'unknown'],
    'weight': ['thin', 'medium', 'fat', 'unknown'],
    'ethnic': ['white', 'black', 'asian', 'india', 'unknown'],
    'haircolor': ['black', 'brown', 'white', 'red', 'gray', 'occluded', 'unknown'],
    'hairstyle': ['bald', 'short', 'medium', 'long', 'horsetail', 'unknown'],
    'beard': ['on', 'off', 'unknown'],
    'moustache': ['on', 'off', 'unknown'],
    'glasses': ['normal', 'sun', 'off', 'unknown'],
    'head': ['hat', 'scarf', 'neckless', 'occluded', 'unknown'],
    'upper': ['tshirt', 'blouse', 'sweater', 'coat', 'bikini', 'naked', 'dress', 'uniform', 'shirt', 'suit', 'hoodie', 'cardiga', 'unknown'],
    'lower': ['jeans', 'leggins', 'pants', 'shorts', 'skirt', 'bikini', 'dress', 'uniform', 'suit', 'unknown'],
    'feet': ['sportshoe', 'classicshoe', 'highheels', 'boots', 'sandals', 'nothing', 'unknown'],
    'bag': ['normal', 'backpack', 'hand', 'rolling', 'umbrella', 'sportif', 'market', 'nothing', 'unknown']
}

class AGReIDv2(BaseImageDataset):
    """
    Dataset statistics:
    # identities:   (+1 for background)
    # images:   (train) +   (query) +   (gallery)
    # id structure:
    """
    dataset_dir = "AGReIDv2"
    dataset_name = 'AGReIDv2'
    logger = logging.getLogger("CLIP-ReID.dataset")

    def __init__(self, root=r'../datasets/', verbose=True, pid_begin=0, **kwargs):
        self.cfg = kwargs.pop('cfg', None)
        super().__init__(**kwargs)
        if self.cfg is None:
            self.use_text = False
            self.use_attr = False
            self.text_prompt = ""
            self.text_format = "individual"
        else:
            self.use_text = self.cfg.MODEL.USE_TEXT
            self.use_attr = self.cfg.MODEL.USE_ATTR
            self.text_prompt = 'X ' * self.cfg.MODEL.TEXT_PROMPT if self.cfg.MODEL.TEXT_PROMPT > 0 else ""
            self.text_format = self.cfg.MODEL.TEXT_FORMAT
            
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        # attr setting loading
        attr_path = osp.join(self.dataset_dir, 'qut_attribute_v8.mat')
        if osp.exists(attr_path):
            attribute_anno, attributes_name = self.generate_attribute_dict(attr_path, "qut_attribute")
            self.delete_attr = []
            self.attribute_map = ATTR_MAP
            self.attribute_num_classes = {k: len(v) for k, v in self.attribute_map.items() if k not in self.delete_attr}
            self.attribute_names = list(self.attribute_num_classes.keys())
            self.attributes = self.parse_attributes(attribute_anno)
            # text setting
            self.texts = self.parse_texts_clean(self.attributes, self.attribute_names, self.attribute_map)
            self.logger.info(f"Attributes: {self.attribute_names}")
        else:
            self.attributes = None
            # text setting
            self.texts = None
            self.attribute_names = []
            self.delete_attr = []
            self.attribute_map = {}
            self.attribute_names = []
            self.attribute_num_classes = {}
            self.attributes = {}
        
        self.train, self.query, self.gallery = self.make_dataset(train_dir=self.train_dir,
                                                                 query_dir=self.query_dir,
                                                                 gallery_dir=self.gallery_dir)

        if verbose:
            # print(f"=> {self.dataset_name} loaded")
            self.logger.info(f"=> {self.dataset_name} loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train, print_cam=True)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query, print_cam=True)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery, print_cam=True)

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')

        return train, query, gallery

    def parse_attributes(self, attributes=None, delete_keys=[]):
        pid_factory = {}
        for pid, attr_label_onehot in attributes.items():
            index = 0
            attr_label_new = []
            for attr in self.attribute_names:
                n = len(self.attribute_map[attr])
                if attr in delete_keys:
                    index += n
                    continue
                segment = attr_label_onehot[index:index + n]
                pos = (segment == 1).nonzero(as_tuple=True)  # 判断1在哪个位置
                # if len(pos[0]) == 0:
                # print(f"Attribute {attr} for pid {pid} has no positive label.")
                label_i = pos[0].item() if len(pos[0]) > 0 else n - 1
                attr_label_new.append(label_i)
                index += n
            pid_factory[pid] = torch.tensor(attr_label_new, dtype=torch.int64)
        return pid_factory

    def parse_texts_clean(self, attributes=None, attribute_names=None, attribute_map=None):

        text_prefix = 'An image of a person with the following attributes: '

        pid_factory = {}
        for pid, attrs_label in attributes.items():
            attr_for_test = ''
            for attr_i, attr_label in enumerate(attrs_label):
                attr_name = attribute_names[attr_i]  # age
                attr_value = attribute_map[attr_name][attr_label]  # ageyoung
                attr_for_test += f"{attr_name.replace(' ', '_')} is {attr_value}, "
            pid_factory[pid] = f'{attr_for_test.strip()[:-1]}.'

        return pid_factory

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, train=False, mode='mix', gallery_pids=None):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_name).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            # pid = int(img_name[1:5])
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2index = {pid: index for index, pid in enumerate(pid_container)}

        dataset = []
        views = {'aerial': 0, 'ground': 0, 'wearable': 0}
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            # camid = int(img_name.split('C')[-1][0])
            # pid = int(img_name[1:5])
            pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_name).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)
            camid, frameid = pattern_camid.search(img_name).groups()
            camid = int(camid)
            timeid = int(pid_part3[-1])
            if camid == 3:   # ground view
                camid = 1
                views['ground'] += 1
                viewid = 0
                if mode == 'aerial' or mode == 'wearable':
                    continue
            elif camid == 2:   # wearable view
                views['wearable'] += 1
                viewid = 1
                if mode == 'aerial' or mode == 'ground':
                    continue
            else:   # aerial view
                camid = 3
                views['aerial'] += 1
                viewid = 2
                if mode == 'wearable' or mode == 'ground':
                    continue

            camid -= 1  # index starts from 0
            pid_idx = pid2index[pid] if train else pid

            if gallery_pids is not None and pid_idx not in gallery_pids:
                continue

            use_text = self.use_text
            if train:
                text_anno = f"A photo of a {self.text_prompt} person with the following attributes: {self.texts[str(pid)]}" if use_text else None
            else:
                text_anno = f"A photo of a {self.text_prompt} person. Attributes are unknown." if use_text else None

            use_attr = self.use_attr and train
            attr_anno = self.attributes[str(pid)] if use_attr else None
            
            # if self.use_text:
            #     platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            #     text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform, capturing natural colors and fine details: {self.get_text_per_img(self.texts, pid, viewid, timeid)}"
            #     dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, text_anno))
            # else:
            #     dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, None))
            dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, [text_anno, attr_anno]))
        self.logger.info(f'{dir_path}: {views}')
        return dataset

    
    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()
        
        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute, key_attribute
    
    
    def get_text_per_img(self, text=None, pid=None, viewid=None, time_id=None):
        return f"the person has attributes labled with {time_id}"


class AGReIDv2_AC(AGReIDv2):
    dataset_dir = "AGReIDv2"
    dataset_name = 'AGReIDv2_ac'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        # query = self._process_dir(query_dir, train=False, mode='aerial')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')

        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)

        return train, query, gallery


class AGReIDv2_CA(AGReIDv2):
    dataset_dir = "AGReIDv2"
    dataset_name = 'AGReIDv2_ca'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        # query = self._process_dir(query_dir, train=False, mode='ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')

        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)

        return train, query, gallery


class AGReIDv2_AW(AGReIDv2):
    dataset_dir = "AGReIDv2"
    dataset_name = 'AGReIDv2_aw'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        # query = self._process_dir(query_dir, train=False, mode='ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='wearable')

        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)

        return train, query, gallery


class AGReIDv2_WA(AGReIDv2):
    dataset_dir = "AGReIDv2"
    dataset_name = 'AGReIDv2_wa'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        # query = self._process_dir(query_dir, train=False, mode='ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')

        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='wearable', gallery_pids=gallery_pids)
        return train, query, gallery


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from utils.logger import setup_logger
    logger = setup_logger("CLIP-ReID", save_dir=None, if_train=True)
    dataset = AGReIDv2()
    dataset_ag = AGReIDv2_AC()
    dataset_ga = AGReIDv2_CA()
    dataset_aa = AGReIDv2_AW()
    dataset_gg = AGReIDv2_WA()
