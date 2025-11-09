
from email.mime import text
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset_aug as ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_AG, ViewBalancedSampler_xhao
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .cp2000 import CP2000_ALL, CP2000_AA, CP2000_AG, CP2000_GA, CP2000_GG, CP2000_AGAG, CP2000_AAGG
from .agreid import AGReID, AGReID_GA, AGReID_AG
from .agreidv2 import AGReIDv2, AGReIDv2_CA, AGReIDv2_AC, AGReIDv2_AW, AGReIDv2_WA
from .cargo import CARGO, CARGO_GA, CARGO_AG, CARGO_AA, CARGO_GG, CARGO_AGAG

import logging
logger = logging.getLogger(__name__)

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,

    'CP2000_ALL': CP2000_ALL,
    'CP2000_GA': CP2000_GA,
    'CP2000_AG': CP2000_AG,
    'CP2000_AA': CP2000_AA,
    'CP2000_GG': CP2000_GG,
    'CP2000_AGAG': CP2000_AGAG,
    'CP2000_AAGG': CP2000_AAGG,

    'AGReID': AGReID,
    'AGReID_GA': AGReID_GA,
    'AGReID_AG': AGReID_AG,

    'AGReIDv2': AGReIDv2,
    'AGReIDv2_CA': AGReIDv2_CA,
    'AGReIDv2_AC': AGReIDv2_AC,
    'AGReIDv2_AW': AGReIDv2_AW,
    'AGReIDv2_WA': AGReIDv2_WA,

    'CARGO': CARGO,
    'CARGO_GA': CARGO_GA,
    'CARGO_AG': CARGO_AG,
    'CARGO_AA': CARGO_AA,
    'CARGO_GG': CARGO_GG,
    'CARGO_AGAG': CARGO_AGAG,
}

__sampler_list = {
    'RandomIdentitySampler': RandomIdentitySampler,
    'RandomIdentitySampler_AG': RandomIdentitySampler_AG,
    'ViewBalancedSampler_xhao': ViewBalancedSampler_xhao,
    'RandomIdentitySampler_DDP': RandomIdentitySampler_DDP,
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, timeids, _, text_tokens, attr = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    timeids = torch.tensor(timeids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    if isinstance(imgs[0], list):
        imgs = list(zip(*imgs))
        imgs = [torch.stack(img_batch, dim=0) for img_batch in imgs]
    text = torch.stack(text_tokens, dim=0) if text_tokens[0] is not None else None
    attr = torch.stack(attr, dim=0) if attr[0] is not None else None
    
    return torch.stack(imgs, dim=0), pids, camids, viewids, timeids, [text, attr]


def val_collate_fn(batch):
    imgs, pids, camids, viewids, timeids, img_paths, text_tokens, attr = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    timeids = torch.tensor(timeids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    if isinstance(imgs[0], list):
        imgs = list(zip(*imgs))
        imgs = [torch.stack(img_batch, dim=0) for img_batch in imgs]
    text = torch.stack(text_tokens, dim=0) if text_tokens[0] is not None else None
    attr = torch.stack(attr, dim=0) if attr[0] is not None else None
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, timeids, img_paths, [text, attr]


def make_dataloader(cfg, dataset_name=None, train=True):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset_name = cfg.DATASETS.NAMES if dataset_name is None else dataset_name
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR, cfg=cfg)
    augment = cfg.DATALOADER.AUG
    train_set = ImageDataset(dataset.train, train_transforms, augment=augment, text_length=cfg.MODEL.TEXT_LEN)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    dataset_dir = dataset.dataset_dir
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            logger.info('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            sampler_name = __sampler_list[cfg.DATALOADER.SAMPLER_NAME]
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=sampler_name(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        logger.info('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        logger.warning('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, augment=augment, text_length=cfg.MODEL.TEXT_LEN)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    ) if len(train_set_normal) > 0 else None
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num, dataset
