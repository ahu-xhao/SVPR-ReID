# coding=utf-8
'''
@Time     : 2023/12/26 09:39:54
@Author   : XHao
@Email    : xhao2510@foxmail.com
'''
# here put the import lib

import os.path as osp

from .bases import BaseImageDataset
from termcolor import colored
import logging
import json
from pathlib import Path
import torch
import copy


class CP2108(BaseImageDataset):
    """
    CP2108
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # images:   (train) +   (query) +   (gallery)
    """
    dataset_dir = 'CP2108'
    dataset_name = 'CP2108'
    logger = logging.getLogger("CLIP-ReID.dataset")

    def __init__(self, root=r'../datasets/', verbose=True, pid_begin=0, **kwargs):
        self.cfg = kwargs.pop('cfg', None)
        super().__init__(**kwargs)

        if self.cfg is None:
            self.use_text = False
            self.use_attr = False
            self.text_prompt = ""
            self.text_type = 'captions'
            self.text_format = 'depart'
            split_version = 100
        else:
            self.use_text = self.cfg.MODEL.USE_TEXT
            self.text_prompt = 'X ' * self.cfg.MODEL.TEXT_PROMPT if self.cfg.MODEL.TEXT_PROMPT > 0 else ""
            self.use_attr = self.cfg.MODEL.USE_ATTR
            self.text_type = self.cfg.MODEL.TEXT_TYPE
            self.text_format = self.cfg.MODEL.TEXT_FORMAT
            split_version = self.cfg.DATASETS.VERSION

        self.split_version = split_version
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'train_v{split_version}.txt')
        self.query_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'query_v{split_version}.txt')
        self.gallery_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'gallery_v{split_version}.txt')
        self.logger.info(colored(f"Dataset split version: {split_version}", 'green'))

        self.use_attrs = ['Glasses', 'Holding Phone', 'Head Accessories', 'Accessories', 'Pose', 'Upper Clothing',
                          'Upper Color', 'Upper Style', 'Lower Clothing', 'Lower Color', 'Lower Style', 'Feet']

        self._check_before_run()
        self.pid_begin = pid_begin

        # attribute setting loading
        if self.use_attr:
            attribute_path = osp.join(self.dataset_dir, 'attributes_labeled_qwen.json')
            attribute_template_path = osp.join(self.dataset_dir, 'attr_translation_idx.json')
            # self.delete_attr = ['View','Illumination']
            self.attribute_anno = json.load(open(attribute_path, 'r'))
            self.attribute_map = json.load(open(attribute_template_path, 'r'))
            self.attribute_num_classes = {k: len(v) for k, v in self.attribute_map.items() if k in self.use_attrs}
            self.attribute_names = sorted(self.attribute_num_classes.keys())
            self.attributes = self.parse_attributes(self.attribute_anno)

        else:
            # self.delete_attr = {}
            self.attribute_anno = {}
            self.attribute_map = {}
            self.attribute_names = []
            self.attribute_num_classes = {}
            self.attributes = {}
        # text setting loading
        if self.use_text:
            if self.text_type == 'captions':
                text_path = osp.join(self.dataset_dir, 'attribute_captions_doubao.json')
                texts = json.load(open(text_path, 'r'))
                self.texts = self.parse_texts(texts)
            else:
                attribute_path = osp.join(self.dataset_dir, 'attributes_labeled_qwen.json')
                attribute_template_path = osp.join(self.dataset_dir, 'attr_translation_idx.json')
                attribute_anno = json.load(open(attribute_path, 'r'))
                attribute_map = json.load(open(attribute_template_path, 'r'))
                attribute_num_classes = {k: len(v) for k, v in attribute_map.items() if k in self.use_attrs}
                attribute_names = sorted(attribute_num_classes.keys())
                self.texts = self.parse_texts_clean(attribute_anno, attribute_names)
        else:
            self.texts = None

        self.train, self.query, self.gallery = self.make_dataset(train_dir=self.train_dir,
                                                                 query_dir=self.query_dir,
                                                                 gallery_dir=self.gallery_dir)

        if verbose:
            self.logger.info(f"=> {self.dataset_name} loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train, print_cam=True)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query, print_cam=True)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery, print_cam=True)

    def parse_texts_clean(self, attribute_anno=None, attribute_names=None):

        pid_factory = {}
        for key, attrs in attribute_anno.items():
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            attr_for_test = ''
            for attr in attribute_names:
                attr_for_test += f"{attr.replace(' ', '_')} is {attrs[attr]}, "
            pid_factory[pid][viewid][timeid] = f'{attr_for_test.strip()[:-1]}.'
            # pid_factory[pid][viewid][timeid] = f'{text_prefix} {attr_for_test.strip()[:-1]}.'

        pid_factory = self.repair_factory(pid_factory)
        return pid_factory

    def parse_texts(self, texts):
        pid_factory = {}
        for key, text in texts.items():
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            pid_factory[pid][viewid][timeid] = text

        pid_factory = self.repair_factory(pid_factory)
        return pid_factory

    def parse_attributes(self, attributes=None, delete_keys=[]):
        pid_factory = {}
        for key, attrs in attributes.items():
            attr_label = []
            for attr in self.attribute_names:
                if attr in delete_keys:
                    continue
                value = attrs.get(attr, 'unknown')
                attr_label.append(self.attribute_map[attr][value])
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            # pid_attributes[key] = torch.tensor(attr_label, dtype=torch.int64)
            pid_factory[pid][viewid][timeid] = torch.tensor(attr_label, dtype=torch.int64)
        pid_factory = self.repair_factory(pid_factory)

        return pid_factory

    def repair_factory(self, pid_factory):
        # fault tolerance
        for pid in pid_factory:
            if 0 not in pid_factory[pid]:
                pid_factory[pid][0] = copy.deepcopy(pid_factory[pid][1])
            elif 1 not in pid_factory[pid]:
                pid_factory[pid][1] = copy.deepcopy(pid_factory[pid][0])

            for viewid in range(2):
                if 0 not in pid_factory[pid][viewid]:
                    pid_factory[pid][viewid][0] = copy.deepcopy(pid_factory[pid][viewid][1])
                elif 1 not in pid_factory[pid][viewid]:
                    pid_factory[pid][viewid][1] = copy.deepcopy(pid_factory[pid][viewid][0])
        return pid_factory

    def get_item_per_img(self, factory, pid, viewid, timeid):
        item = factory[pid][viewid][timeid]
        return item

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        return train, query, gallery

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
        # img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        root = Path(dir_path)
        split_version = f"images_v{self.split_version}" if self.split_version >= 100 else 'images'
        data_dir = root.parent.parent / split_version
        with open(root, 'r') as f:
            data = f.read().strip().split('\n')
        img_paths = [str(data_dir / line) for line in data]

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            try:
                pid = int(img_name.split('_')[0])
            except Exception as e:
                self.logger.error(f"Error parsing PID from image name '{img_name}': {e}")
                raise
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2index = {pid: index for index, pid in enumerate(pid_container)}
        dataset = []
        views = {'aerial': 0, 'ground': 0}
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            parts = img_name.split('_')
            pid = int(parts[0])
            camid = int(parts[1])
            timeid = int(parts[2])
            sceneid = int(parts[3])
            if camid < 23:
                views['ground'] += 1
                viewid = 0  # ground view
                if mode == 'aerial':
                    continue
                if mode == 'aerial-ground':
                    camid = 1
            else:   # aerial view
                views['aerial'] += 1
                viewid = 1
                if mode == 'ground':
                    continue
                if mode == 'aerial-ground':
                    camid = 2

            camid -= 1  # index starts from 0
            pid_idx = pid2index[pid] if train else pid

            if gallery_pids is not None and pid_idx not in gallery_pids:
                continue

            use_text = self.use_text
            if train:
                text_anno = f"A photo of a {self.text_prompt} person with the following attributes: {self.get_item_per_img(self.texts, pid, 0, 1)}" if use_text else None
            else:
                text_anno = f"A photo of a {self.text_prompt} person." if use_text else None

            use_attr = self.use_attr
            attr_anno = self.get_item_per_img(self.attributes, pid, 0, 1) if use_attr else None

            dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, [text_anno, attr_anno]))

        self.logger.info(f'{dir_path}: {views}')
        return dataset


class CP2108_ALL(CP2108):
    dataset_name = 'CP2108_all'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)

        return train, query, gallery


class CP2108_GA(CP2108):
    dataset_name = 'CP2108_ga'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)

        return train, query, gallery


class CP2108_AG(CP2108):
    dataset_name = 'CP2108_ag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        return train, query, gallery


class CP2108_AA(CP2108):
    dataset_dir = 'CP2108'
    dataset_name = 'CP2108_aa'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'quer')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)

        return train, query, gallery


class CP2108_GG(CP2108):
    dataset_dir = 'CP2108'
    dataset_name = 'CP2108_gg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)

        return train, query, gallery


class CP2108_AGAG(CP2108):
    dataset_dir = 'CP2108'
    dataset_name = 'CP2108_agag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):

        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='aerial-ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial-ground')

        return train, query, gallery


class CP2108_AAGG(CP2108):
    dataset_name = 'CP2108_aagg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)

        return train, query, gallery


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    print(f"{os.getcwd()}")
    from utils.logger import setup_logger
    logger = setup_logger("CLIP-ReID", save_dir=None, if_train=True)
    dataset = CP2108_ALL()
    dataset_ag = CP2108_AG()
    dataset_ga = CP2108_GA()
    dataset_aa = CP2108_AA()
    dataset_gg = CP2108_GG()
    dataset_agag = CP2108_AGAG()
