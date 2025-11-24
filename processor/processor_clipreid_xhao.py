# coding=utf-8
'''
@Time     : 2025/04/22 04:00:53
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib
# officical packages
import logging
import os
from termcolor import colored
# torch packages
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
# local packages
from utils.meter import AverageMeter
from utils.metrics_xhao import R1_mAP_eval
from loss.supcontrast import SupConLoss
from utils.comm import dict_to_table


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
    log_period = iters_per_ehoch // (cfg.SOLVER.LOG_PERIOD - 1)

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
        for n_iter, (img, vid, target_cam, target_view, target_time) in enumerate(train_loader_stage1):
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

    best_model_info = {'dataset': '', 'epoch': 0, 'mAp': 0, 'mINP': 0, 'rank1': 0, 'rank5': 0}
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

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0 or i == iters_per_ehoch - 1:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), iters_per_ehoch,
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


def do_train(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    # log_period = cfg.SOLVER.LOG_PERIOD
    # for n_iter, _ in enumerate(train_loader_stage2):
    #     pass
    iters_per_ehoch = len(train_loader_stage2) + 1
    log_period = iters_per_ehoch // (cfg.SOLVER.LOG_PERIOD - 1)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    # instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("CLIP-ReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    otherloss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    scaler = amp.GradScaler()
    # xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    # with torch.no_grad():
    #     for i in range(i_ter):
    #         if i + 1 != i_ter:
    #             l_list = torch.arange(i * batch, (i + 1) * batch)
    #         else:
    #             l_list = torch.arange(i * batch, num_classes)
    #         with amp.autocast(enabled=True):
    #             text_feature = model(label=l_list, get_text=True)
    #         text_features.append(text_feature.cpu())
    #     text_features = torch.cat(text_features, 0).cuda()
    best_model_info = {'epoch': 0, 'mAp': 0, 'mINP': 0, 'rank1': 0, 'rank5': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        otherloss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        model.train()
        for n_iter, (img, vid, target_cam, target_view, target_time, text) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            target_time = target_time.to(device)
            if text is not None:
                text = [t.to(device) if t is not None else None for t in text]

            with amp.autocast(enabled=True):
                model_name = cfg.MODEL.ARCH_NAME
                if model_name in ['TransReID', 'ViT']:
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    loss = loss_fn(score, feat, target, target_cam, add_type='other')
                    logits = score[0]
                    loss_add = torch.tensor(0.)

                elif model_name in ['VDT', 'VDT_CVPR']:
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    loss = loss_fn(score[:-1], feat[:-1], target, target_cam)
                    logits = score[0]
                    # loss_view = loss_fn.get_id_loss(score[-1:], None, target_view)
                    loss_view = loss_fn.get_view_loss(score[-1], feat[-1], feat[0], target_view, view_lambda=0.1)
                    loss_add = torch.tensor(0.) if len(feat[:-1]) == 1 else loss_fn.get_pts_loss(feat[:-1])
                    loss = loss + loss_view * 1.0 + loss_add * 0.1

                elif model_name in ['CLIP_View']:
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    loss = loss_fn(score[:-1], feat[:-1], target, target_cam)
                    logits = score[0]
                    loss_add = loss_fn.get_view_loss(score[-1], feat[-1], feat[0], target_view)
                    loss_add += torch.tensor(0.) if len(feat[:-1]) == 1 else loss_fn.get_pts_loss(feat[:-1])
                    loss = loss + loss_add * 1.0

                elif model_name == 'SeCap':
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    loss = loss_fn(score[:-1], feat[:-1], target, target_cam)
                    logits = score[0]
                    loss_add = loss_fn.get_view_loss(score[-1], feat[-1], feat[0], target_view, view_lambda=0.1)
                    loss = loss + loss_add

                elif model_name == 'CLIP_view_attrText_cvpr_local':
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time, text=text)
                    logits = score[0]
                    loss = loss_fn(score[:-2], [*feat[:-3], feat[-2]], target, target_cam, add_type='other')
                    loss_attr = score[-2]
                    loss_view = loss_fn.get_view_loss(score[-1], feat[-1], feat[0], target_view, view_lambda=1.)
                    # loss_add = torch.tensor(0.) if len(feat[:-1]) == 1 else loss_fn.get_pts_loss(feat[:-1])
                    loss_add = torch.tensor(0.)  # attr loss
                    loss = loss + loss_view + loss_add * 0.05 + loss_attr * 0.05

                elif model_name in ['CLIP', 'CLIP_prompt', 'CLIP_GCA']:
                    # ~~~~~~~ CLIP-ReID ~~~~~~~~
                    score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    logits = image_features @  text_features.t()
                    loss = loss_fn(score, feat, target, target_cam, i2tscore=logits)
                    loss_add = torch.tensor(0.)

                elif model_name in ['CLIP_baseline', 'CLIP_text', 'CLIP_text_ca', 'CLIP_Attr']:
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time, text=text)
                    # logits = image_features @  text_features.t()
                    logits = score[0]
                    # logits = torch.stack(score, dim=1).mean(dim=1)
                    loss = loss_fn(score, feat, target, target_cam)
                    loss_add = torch.tensor(0.)

                elif model_name in ['CLIP_CVPR']:
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time, text=text)
                    logits = score[0]
                    loss = loss_fn(score, feat, target, target_cam)

                    loss_add = loss_fn.get_pts_loss(feat[:-1])
                    loss = loss + loss_add * 0.1
                elif model_name == 'CLIP_inverseText_localPatchs_CVPR':
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time, text=text)
                    # logits = image_features @  text_features.t()
                    # logits = torch.mean(torch.stack(score), dim=1)
                    logits = score[0]
                    loss = loss_fn(score, feat, target, target_cam, add_type='other')
                    loss_add = loss_fn.get_pts_loss(feat[:-1]) * 0.1
                    loss += loss_add

                elif model_name == 'CLIP_text_prompt_localPatchs_CVPR':
                    score, feat = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time, text=text)
                    logits = score[0]
                    loss = loss_fn(score, feat, target, target_cam, add_type='other')
                    # loss_add = loss_fn.get_pts_loss(feat[:-1])
                    # loss += loss_add * 0.1
                    loss_add = torch.tensor(0.)
                    # loss = loss

                elif model_name == 'CLIP_GLView':
                    score, feat, image_features, loss_add = model(
                        x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time,
                        # text_feats=text_features
                    )
                    logits = image_features @  text_features.t()
                    loss = loss_fn(score, feat, target, target_cam, i2tscore=logits)
                    loss += loss_add

                elif model_name == 'CLIP_PTS':
                    score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    logits = image_features @  text_features.t()
                    loss = loss_fn(score[:2], feat[:3], target, target_cam, i2tscore=logits)
                    loss_pts_id = loss_fn.get_id_loss(score[2:], feat[3:], target, add_type='mean')
                    loss_pts_tri = loss_fn.get_triplet_loss(score[2:], feat[3:], target, add_type='mean')
                    loss_pts_feat = loss_fn.get_pts_loss([feat[:, 1], feat[:, -3], feat[:, -2], feat[:, -1]])
                    pts_weight = 0.1
                    loss_add = loss_pts_id * pts_weight + loss_pts_tri * pts_weight + loss_pts_feat * pts_weight

                    loss += loss_add

                elif model_name == 'CLIP_AUG':
                    score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    logits = image_features @  text_features.t()
                    loss = loss_fn(score, feat, target, target_cam, i2tscore=logits)
                    # loss_aug_id = loss_fn.get_id_loss(score[2:], feat[2:], target)
                    # loss_aug_tri = loss_fn.get_triplet_loss(score[2:], feat[2:], target)
                    # loss_add = loss_aug_id + loss_aug_tri
                    # loss += loss_add
                    loss_add = torch.tensor(0.)
                elif model_name == 'CLIP_LATEX':
                    score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view, time_label=target_time)
                    text_features = feat[-1]
                    loss = loss_fn(score, feat, target, target_cam)
                    loss_add = nn.MSELoss()(image_features, text_features)
                    # loss_aug_id = loss_fn.get_id_loss(score[2:], feat[2:], target)
                    # loss_aug_tri = loss_fn.get_triplet_loss(score[2:], feat[2:], target)
                    # loss_add = loss_aug_id + loss_aug_tri
                    loss += loss_add
                    logits = score[0]
                    # loss_add = torch.tensor(0.)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            otherloss_meter.update(loss_add.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0 or n_iter == iters_per_ehoch - 1:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (other: {:.3f}), Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), iters_per_ehoch,
                                    loss_meter.avg, otherloss_meter.avg,
                                    acc_meter.avg, scheduler.get_lr()[0]))

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {}(total {} iters) done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, n_iter, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute(cross_time=cfg.TEST.CROSS_TIME)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.2%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, viewids, timeids, imgpath, text) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = viewids.to(device)
                        target_time = timeids.to(device)
                        if text is not None:
                            text = [t.to(device) if t is not None else None for t in text]
                        feat = model(img, cam_label=camids, view_label=target_view, time_label=target_time, text=text)
                        # evaluator.update((feat, vid, camid, timeids))
                        evaluator.update((feat, vid, camid, viewids, timeids, imgpath, img))
                # cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
                time_mode = cfg.TEST.TIME_MODE
                cmc, mAP, mINP, _, _, _, _, _ = evaluator.new_compute(time_mode=time_mode, visualizer=None)
                table_data = {'mAP': mAP, 'mINP': mINP}
                for r in [1, 5, 10]:
                    # logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                    table_data[f'rank{r}'] = cmc[r - 1]
                logger.info(f"Validation Results - Epoch: {epoch}\n" + dict_to_table(table_data, use_color=False))

                if mAP > best_model_info['mAp']:
                    best_model_info['epoch'] = epoch
                    best_model_info['mAp'] = mAP
                    best_model_info['mINP'] = mINP
                    best_model_info['rank1'] = cmc[0]
                    best_model_info['rank5'] = cmc[4]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'transformer_best.pth'))
                # logger.info('Best result:\n' + dict_to_table(best_model_info,use_color=True))
                best_data = f"epoch: {best_model_info['epoch']}, " + ", ".join([f'{k}: {v:.2%}' for k, v in best_model_info.items() if k != 'epoch'])
                logger.info(colored(f'Best Result: {best_data}', "green"))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))


def do_inference(cfg,
                 dataset_name,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("CLIP-ReID.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg, is_training=False)
    evaluator.reset()

    use_grad = False

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, pid, camid, camids, viewids, timeids, imgpath, text) in enumerate(val_loader):
        diy_context = torch.cuda.amp.autocast if use_grad else torch.no_grad
        with diy_context():
            img = img.to(device)
            camids = camids.to(device)
            target_view = viewids.to(device)
            target_time = timeids.to(device)
            if text is not None:
                text = [t.to(device) if t is not None else None for t in text]
            score, feat = model(img, cam_label=camids, view_label=target_view, time_label=target_time, text=text, test_score=True)
            evaluator.update((feat, pid, camid, viewids, timeids, imgpath, img))
        # visualize Grad-CAM
        # inputs = {'images': img, 'img_paths': imgpath}
        # outputs = feat
        # cam_save_path = Path(f'grad_cam/{cfg.MODEL.ARCH_NAME}/{dataset_name}')
        # _ = visualizer.vis_grad_cam(inputs, outputs, use_grad=use_grad, cam_save_path=cam_save_path)

        # cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute(cross_time=True, visualizer=visualizer)
    time_mode = cfg.TEST.TIME_MODE
    if dataset_name.endswith('AGAG'):
        view_mode = 'cross'
    elif dataset_name.endswith('AAGG'):
        view_mode = 'same'
    else:
        view_mode = 'mix'
    cmc, mAP, mINP, _, _, _, _, _ = evaluator.new_compute(time_mode=time_mode, view_mode=view_mode, visualizer=None)

    # logger.info("Validation Results ")
    # logger.info("mAP: {:.2%}".format(mAP))
    table_data = {'mAP': mAP, 'mINP': mINP}
    for r in [1, 5, 10]:
        # logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        table_data[f'rank{r}'] = cmc[r - 1]
    logger.info(f"Validation Results:\n" + dict_to_table(table_data, use_color=False))

    return table_data
