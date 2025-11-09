import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from config import cfg
import argparse
from datasets.make_dataloader_xhao import make_dataloader
from model import make_model
from processor.processor_clipreid_xhao import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/CP2000/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("CLIP-ReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 测试模型
    for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, dataset = make_dataloader(cfg, dataset_name)
        cfg.defrost()
        cfg.DATASETS.DATASET_DIR = dataset.dataset_dir
        cfg.DATASETS.ATTR_NAME = dataset.attribute_names
        cfg.DATASETS.ATTR_CLASS = [dataset.attribute_num_classes[i] for i in dataset.attribute_names]
        cfg.freeze()
        if idx == 0:
            model = make_model(cfg, num_class=num_classes, camera_num=39, view_num=view_num)
            model.load_param(cfg.TEST.WEIGHT)
        do_inference(cfg,
                     dataset_name,
                     model,
                     val_loader,
                     num_query)
