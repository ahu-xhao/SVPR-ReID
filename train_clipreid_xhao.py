
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.logger import setup_logger
from datasets.make_dataloader_xhao import make_dataloader
from model import make_model
from processor.processor_clipreid_xhao import do_train_stage1, do_train, do_inference
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage, make_optimizer
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import ReIDLoss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/CP2000/vit_clipreid_baseline.yml", help="path to config file", type=str
        # "--config_file", default="configs/CP2000/vit_clipreid_svpr.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("CLIP-ReID", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num, dataset = make_dataloader(cfg)
    cfg.defrost()
    cfg.DATASETS.DATASET_DIR = dataset.dataset_dir
    cfg.DATASETS.ATTR_NAME = dataset.attribute_names
    cfg.DATASETS.ATTR_CLASS = [dataset.attribute_num_classes[i] for i in dataset.attribute_names]
    cfg.freeze()
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    loss_func = ReIDLoss(cfg, num_classes=num_classes)
    center_criterion = loss_func.center_criterion

    if cfg.SOLVER.STAGE1.TRAIN:
        optimizer_1stage = make_optimizer_1stage(cfg, model)
        scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1.LR_MIN,
                                            warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range=None)
        do_train_stage1(
            cfg,
            model,
            train_loader_stage1,
            optimizer_1stage,
            scheduler_1stage,
            args.local_rank
        )
    elif cfg.MODEL.ARCH_NAME == 'CLIP_ReID':
        model.load_param_text(cfg.SOLVER.STAGE1.PRETRAIN_PATH)
        # model.load_param(cfg.SOLVER.STAGE1.PRETRAIN_PATH)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )

    # 测试模型
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, dataset = make_dataloader(cfg, dataset_name)
        cfg.defrost()
        cfg.DATASETS.DATASET_DIR = dataset.dataset_dir
        cfg.DATASETS.ATTR_NAME = dataset.attribute_names
        cfg.DATASETS.ATTR_CLASS = [dataset.attribute_num_classes[i] for i in dataset.attribute_names]
        cfg.freeze()
        if idx == 0:
            # model = make_model(cfg, num_class=num_classes, camera_num=39, view_num=view_num)
            model.load_param(cfg.TEST.WEIGHT)
        do_inference(cfg,
                     dataset_name,
                     model,
                     val_loader,
                     num_query)
