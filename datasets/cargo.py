# coding=utf-8
'''
@Time     : 2023/12/26 09:39:54
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import glob
import json
import os.path as osp
from .bases import BaseImageDataset


class CARGO(BaseImageDataset):
    """
    CARGO
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities:   (+1 for background)
    # images:   (train) +   (query) +   (gallery)
    # id structure:     
    """
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO'

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
        
        self.attribute_anno = {}
        self.attribute_map = {}
        self.attribute_names = []
        self.attribute_num_classes = {}
        self.attributes = {}

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
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
        img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[2])
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2index = {pid: index for index, pid in enumerate(pid_container)}

        dataset = []
        views = {'aerial': 0, 'ground': 0}
        camids = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            parts = img_name.split('_')
            pid = int(parts[2])
            camid = int(parts[0][3:])
            timeid = {'day': 1, 'night': 0}[parts[1]]
            sceneid = -1
            if camid > 5:
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

            # if self.use_text:
            #     platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            #     text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform, capturing natural colors and fine details: {self.get_text_per_img(self.texts, pid, viewid, timeid)}"
            #     dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, text_anno))
            # else:
            #     dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, None))
            use_text = self.use_text
            text_anno = f"A photo of a {self.text_prompt} person. Attributes are not available." if use_text else None
            use_attr = self.use_attr
            attr_anno = None if use_attr else None
            dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, [text_anno, attr_anno]))
        self.logger.info(f'{dir_path}: {views}')
        # self.logger.info(f"{camids}")
        return dataset

    def get_text_per_img(self, text=None, pid=None, viewid=None, time_id=None):
        return f"the person has attributes labled with {time_id}"


class CARGO_GA(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_ga'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)

        return train, query, gallery


class CARGO_AG(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_ag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        return train, query, gallery


class CARGO_AA(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_aa'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)

        return train, query, gallery


class CARGO_GG(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_gg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)

        return train, query, gallery


class CARGO_AGAG(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_agag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):

        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='aerial-ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial-ground')

        return train, query, gallery


class CARGO_AAGG(CARGO):
    dataset_dir = 'CARGO'
    dataset_name = 'CARGO_aagg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):

        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')

        return train, query, gallery


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from utils.logger import setup_logger
    logger = setup_logger("CLIP-ReID", save_dir=None, if_train=True)
    dataset = CARGO()
    dataset_ag = CARGO_AG()
    dataset_ga = CARGO_GA()
    dataset_aa = CARGO_AA()
    dataset_gg = CARGO_GG()
    dataset_agag = CARGO_AGAG()
