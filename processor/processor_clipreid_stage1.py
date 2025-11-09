import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    # log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    # for n_iter, _ in enumerate(train_loader_stage1):
    #     pass
    iters_per_ehoch = len(train_loader_stage1) + 1
    log_period = iters_per_ehoch // (cfg.SOLVER.STAGE1.LOG_PERIOD - 1)

    logger = logging.getLogger("CLIP-ReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    views = []
    times = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view, target_time, captions) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            views_t = target_view.to(device)
            times_t = target_time.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                for i, v, t, img_feat in zip(target, views_t, times_t, image_feature):
                    labels.append(i)
                    views.append(v)
                    times.append(t)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda()  # N
        views_list = torch.stack(views, dim=0).cuda()  # N
        times_list = torch.stack(times, dim=0).cuda()  # N

        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, views, times, image_features

    best_model_info = {'mAp': 0, 'rank1': 0, 'epoch': 0}
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]
            view = views_list[b_list]
            time_ = times_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label=target, view_label=view, time_label=time_, get_text=True)
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            loss = loss_i2t + loss_t2i

            # loss_i2t_view = xent(image_features, text_features, view, view)
            # loss_t2i_view = xent(text_features, image_features, view, view)
            # loss_i2t_time = xent(image_features, text_features, time_, time_)
            # loss_t2i_time = xent(text_features, image_features, time_, time_)
            # loss_view_time = loss_i2t_view + loss_t2i_view + loss_i2t_time + loss_t2i_time
            # loss = loss + loss_view_time

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0 or i == len(train_loader_stage1) - 1:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            # text_related_weights = {k: v for k, v in model.state_dict().items() if k.startswith('prompt_learner') or k.startswith('text_encoder')}
            text_related_weights = {k: v for k, v in model.state_dict().items()}

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(text_related_weights,
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(text_related_weights,
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
