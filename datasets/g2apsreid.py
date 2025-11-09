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
sys.path.append(os.getcwd())
import os.path as osp
from .bases import BaseImageDataset
import re
import logging

__all__ = ['AGReIDv2', ]


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
            self.text_prompt = ""
        else:
            self.use_text = self.cfg.MODEL.USE_TEXT
            self.text_prompt = 'X ' * self.cfg.MODEL.TEXT_PROMPT if self.cfg.MODEL.TEXT_PROMPT > 0 else ""
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_all')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        # text setting loading
        text_path = osp.join(self.dataset_dir, 'attribute_captions_doubao.json')
        if osp.exists(text_path):
            with open(text_path, 'r') as f:
                self.texts = json.load(f)
        else:
            self.texts = None
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
                camid = 1
                views['aerial'] += 1
                viewid = 2
                if mode == 'wearable' or mode == 'ground':
                    continue

            camid -= 1  # index starts from 0
            pid_idx = pid2index[pid] if train else pid

            if gallery_pids is not None and pid_idx not in gallery_pids:
                continue

            if self.use_text:
                platform_name = {0: "UAV", 1: "CCTV", 2: "wearable"}[viewid]
                text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform, capturing natural colors and fine details: {self.get_text_per_img(self.texts, pid, viewid, timeid)}"
                dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, text_anno))
            else:
                dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, None))
        self.logger.info(f'{dir_path}: {views}')
        return dataset

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
