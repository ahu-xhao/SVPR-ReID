import os
import random
import sys
sys.path.insert(0, os.getcwd())

import logging
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from utils.simple_tokenizer import SimpleTokenizer,tokenize
import os.path as osp
from pathlib import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    logger = logging.getLogger("CLIP-ReID.dataset")

    def get_imagedata_info(self, data, print_cam=False):
        pids, cams, tracks = [], [], []
        timeids = []

        for _, pid, camid, trackid, timeid, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            timeids += [timeid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        times = set(timeids)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        num_times = len(times)
        if print_cam:
            self.logger.info(f"camera label is :{cams}")
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        self.logger.info("Dataset statistics:")
        self.logger.info("  ----------------------------------------")
        self.logger.info("  subset   | # ids | # images | # cameras |")
        self.logger.info("  ----------------------------------------")
        self.logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        self.logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        self.logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        self.logger.info("  ----------------------------------------")

        # print("Dataset statistics:")
        # print("  ----------------------------------------")
        # print("  subset   | # ids | # images | # cameras")
        # print("  ----------------------------------------")
        # print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        # print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        # print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        # print("  ----------------------------------------")





class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, text_length: int = 77, truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if len(self.dataset[index]) == 5:  # use timeid
            img_path, pid, camid, trackid, timeid = self.dataset[index]
            img = read_image(img_path)
            if self.transform:
                img = self.transform(img)
                return img, pid, camid, trackid, timeid, img_path.split('/')[-1]
        elif len(self.dataset[index]) == 6:  # use timeid and text
            img_path, pid, camid, trackid, timeid, text = self.dataset[index]
            img = read_image(img_path)
            text_tokens = tokenize(text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) if text else None
            if self.transform:
                img = self.transform(img)
                return img, pid, camid, trackid, timeid, img_path.split('/')[-1], text_tokens
        else:  # origin
            img_path, pid, camid, trackid = self.dataset[index]
            img = read_image(img_path)
            if self.transform:
                img = self.transform(img)
                return img, pid, camid, trackid, img_path.split('/')[-1]


from .data_aug import random_rotate_image, style_transfer


class ImageDataset_aug(Dataset):
    def __init__(self, dataset, transform=None, augment=False, text_length: int = 77, truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.augment = augment
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.sentence = False
        self.max_vocab = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem2__(self, index):
        if len(self.dataset[index]) == 5:
            img_path, pid, camid, viewid, timeid = self.dataset[index]
            text_tokens = None
        elif len(self.dataset[index]) == 6:  # use timeid and text
            img_path, pid, camid, viewid, timeid, text_attr = self.dataset[index]
            text, attr = text[0], text[1]
            if text:
                if self.sentence:
                    sentences = text.split(".")
                    text_tokens = [tokenize(s, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for s in sentences]
                else:
                    text_tokens = tokenize(text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            else:
                text_tokens = None
        else:
            img_path, pid, camid, viewid = self.dataset[index]
            timeid = 1
            text_tokens = None
            attr = None
        
        img = read_image(img_path)
        if self.augment:
            if timeid == 0:
                img_aug = style_transfer(img, brightness=1.8, contrast=1.5, color=0.2, sharpness=0.5)
            else:
                img_aug = style_transfer(img, brightness=0.2, contrast=1.5, color=0.2, sharpness=0.5)
            img_aug = random_rotate_image(img_aug, -90, 90)
            if self.transform:
                img = self.transform(img)
                img_aug = self.transform(img_aug)
            return [img, img_aug], pid, camid, viewid, timeid, img_path.split('/')[-1], text_tokens

        else:
            if self.transform:
                img = self.transform(img)
            return img, pid, camid, viewid, timeid, img_path.split('/')[-1], text_tokens

    def __getitem__(self, index):
        if len(self.dataset[index]) == 5:
            img_path, pid, camid, viewid, timeid = self.dataset[index]
            text_tokens = None
            attr = None
        elif len(self.dataset[index]) == 6:  # use timeid and text
            img_path, pid, camid, viewid, timeid, text_attr = self.dataset[index]
            text, attr = text_attr[0], text_attr[1]
            if text is not None:
                # if self.sentence:
                #     sentences = text.split(".")
                #     text_tokens = [tokenize(s, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) for s in sentences]
                # else:
                #     text_tokens = tokenize(text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
                text_tokens = tokenize(text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            else:
                text_tokens = None
            # text_tokens = text
        else:
            img_path, pid, camid, viewid = self.dataset[index]
            timeid = 1
            text_tokens = None
            attr = None
        img = read_image(img_path)
        if self.augment:
            if random.random() < 0.5:
                if timeid == 0:
                    img = style_transfer(img, brightness=1.8, contrast=1.5, color=0.2, sharpness=0.5)
                else:
                    img = style_transfer(img, brightness=0.2, contrast=1.5, color=0.2, sharpness=0.5)
            # img_aug = random_rotate_image(img_aug, -90, 90)
        if self.transform:
            img = self.transform(img)
        return img, pid, camid, viewid, timeid, img_path.split('/')[-1], text_tokens, attr
