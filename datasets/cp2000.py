# coding=utf-8
'''
@Time     : 2023/12/26 09:39:54
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import glob
import os.path as osp

from .bases import BaseImageDataset
from termcolor import colored
import logging
import json
from pathlib import Path
import torch
import copy


final_div = {1536, 1538, 1541, 2054, 2055, 2057, 2058, 11, 1549, 1550, 13, 1552, 2062, 1551, 1555, 17, 2064, 1559, 24, 1561, 1563, 1564, 29, 1566, 1567, 31, 33, 1572, 1573, 2084, 1575, 1576, 2086, 1578, 39, 1577, 1580, 2094, 47, 1585, 49, 2045, 1588, 2101, 1589, 2103, 1591, 1593, 1592, 2107, 1612, 1614, 1619, 1622, 1625, 1631, 1633, 1637, 1641, 1149, 1156, 1162, 1163, 1164, 1682, 1702, 1197, 1198, 1709, 1201, 1714, 1203, 1206, 1211, 1724, 1730, 1732, 1226, 1229, 1232, 1748, 1246, 1247, 1250, 1764, 1253, 1766, 1768, 1257, 1261, 1262, 1774, 1269, 1781, 1271, 1270, 1273, 1274, 1276, 1279, 1281, 1287, 1800, 1289, 1291, 1295, 1298, 1299, 1300, 276, 1302, 1306, 282, 283, 1320,
             1323, 1326, 1327, 1841, 1341, 1348, 1860, 1865, 1356, 1357, 1360, 1362, 1369, 1372, 1374, 1376, 1378, 1379, 1380, 1389, 1390, 1906, 1395, 1396, 1909, 1913, 1914, 1401, 1404, 1402, 1411, 1415, 1416, 1417, 1418, 1420, 1422, 1423, 1424, 1425, 1935, 1428, 1430, 1432, 1433, 1945, 1434, 1438, 1439, 1444, 1957, 1958, 1449, 1450, 1961, 1452, 1453, 1454, 1455, 1451, 1969, 1457, 1458, 1968, 1971, 1974, 1972, 1464, 1463, 1979, 1981, 1470, 1471, 1472, 1985, 1986, 451, 1478, 1990, 1480, 1481, 1994, 1483, 1998, 2001, 2005, 1493, 1495, 1498, 1499, 2010, 2015, 1505, 2018, 1507, 2021, 998, 2023, 1510, 1514, 1515, 1518, 1519, 1520, 2031, 2034, 2035, 1521, 2037, 1525, 1528, 2041, 2042, 1533, 1535}


class CP2000(BaseImageDataset):
    """
    CP2000
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities:   (+1 for background)
    # images:   (train) +   (query) +   (gallery)
    # id structure:
    """
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000'
    logger = logging.getLogger("CLIP-ReID.dataset")

    def __init__(self, root=r'../datasets/', verbose=True, pid_begin=0, **kwargs):
        self.cfg = kwargs.pop('cfg', None)
        super().__init__(**kwargs)

        if self.cfg is None:
            self.use_text = False
            self.use_attr = False
            self.text_prompt = ""
            self.text_type = 'captions'
            self.text_format = 'hybird'
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
                # self.logger.info(colored(f"Attributes ({len(attribute_names)}): {attribute_names}", 'green'))
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

        text_prefix = 'An image of a person with the following attributes: '

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
                # if value =="Sweatshirts":
                #     value = "jacket"
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
        # 容错
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
        # key = f"{pid}_{viewid}_{timeid}"
        # text = texts.get(key, None)
        # if text is not None:
        #     key_brother = f"{pid}_{viewid}_{0 if timeid == 1 else 1}"
        #     text = texts.get(key_brother, None)
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
            # if pid in final_div:
            #     continue
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

            # if pid in v501_easy[:150]:
            #     continue
            # platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            # platform_name = "X"
            use_text = self.use_text
            if train:
                text_anno = f"A photo of a {self.text_prompt} person with the following attributes: {self.get_item_per_img(self.texts, pid, 0, 1)}" if use_text else None
            else:
                text_anno = f"A photo of a {self.text_prompt} person." if use_text else None

            use_attr = self.use_attr
            # attr_anno = self.get_item_per_img(self.attributes, pid, 0, 1) if use_attr else None
            attr_anno = None if use_attr else None

            # if self.use_text:
            #     platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            #     if train:
            #         text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform, capturing natural colors and fine details: {self.get_item_per_img(self.texts, pid, 0, 1)}"
            #         attr_anno = self.get_item_per_img(self.attributes, pid, viewid, 1) if self.use_attr else None
            #     else:
            #         text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform. capturing natural colors and fine details are unkown."
            #         attr_anno = None
            # else:
            #     text_anno = None
            #     if train:
            #         attr_anno = self.get_item_per_img(self.attributes, pid, 0, 1) if self.use_attr else None
            #     else:
            #         attr_anno = None
            dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, [text_anno, attr_anno]))

        self.logger.info(f'{dir_path}: {views}')
        return dataset


class CP2000_ALL(CP2000):
    dataset_name = 'CP2000_all'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)
        # query_new = [entry for entry in query]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


class CP2000_GA(CP2000):
    dataset_name = 'CP2000_ga'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in final_div]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')

        return train, query, gallery


class CP2000_AG(CP2000):
    dataset_name = 'CP2000_ag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'query')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


class CP2000_AA(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_aa'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'quer')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        # query_new = query
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


class CP2000_GG(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_gg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'query')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)
        # query_new = [entry for entry in query]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


class CP2000_AGAG(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_agag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):

        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='aerial-ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial-ground')

        # query_new = [entry for entry in query if entry[1] not in cross_view_me_easy]
        # query_new = query
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


class CP2000_AAGG(CP2000):
    dataset_name = 'CP2000_aagg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in final_div]
        return train, query, gallery


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    print(f"{os.getcwd()}")
    from utils.logger import setup_logger
    logger = setup_logger("CLIP-ReID", save_dir=None, if_train=True)
    dataset = CP2000_ALL()
    dataset_ag = CP2000_AG()
    dataset_ga = CP2000_GA()
    dataset_aa = CP2000_AA()
    dataset_gg = CP2000_GG()
    dataset_agag = CP2000_AGAG()
