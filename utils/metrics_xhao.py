import torch
import numpy as np
from utils.reranking import re_ranking
import logging
from utils.visualize import Visualizer_DiY
from collections import defaultdict


logger = logging.getLogger('SVPR-ReID.EVAL')


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2,)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_timeids=None, g_timeids=None, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    # num_valid_q = 0.  # number of valid query
    gallery_keep_indices = []
    query_keep_indices = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        if q_timeids is not None and g_timeids is not None:
            remove_time = (g_timeids[order] == q_timeids[q_idx])
            remove = remove | remove_time
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        # num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        query_keep_indices.append(q_idx)
        gallery_keep_indices.append(keep)

    num_valid_q = len(query_keep_indices)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    valid_qids = list(np.unique(q_pids[query_keep_indices]))
    logger.info(f'num_valid_q: {num_valid_q}; valid query pids:({len(valid_qids)})')

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    # mAP = np.mean(all_AP)
    # mINP = np.mean(all_INP)

    return all_cmc, all_AP, all_INP, query_keep_indices, gallery_keep_indices


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, cfg=None, is_training=True):
        super().__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.cfg = cfg
        self.is_training = is_training
        self.reset()

    def reset(self):
        self.feats = []
        self.images = []
        self.pids = []
        self.camids = []
        self.timeids = []
        self.viewids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output[:3]
        # timeid = None if len(output) == 3 else output[-1]
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        viewid, timeid, img_path, images = output[3:7] if len(output) == 7 else (None, None, None, None)
        images = images.cpu() if images is not None else None

        if timeid is not None:
            self.timeids.extend(np.asarray(timeid))
        if viewid is not None:
            self.viewids.extend(np.asarray(viewid))
        if img_path is not None:
            self.img_paths.extend(img_path)
        if images is not None:
            self.images.extend(images)

    def compute(self, cross_time=False, visualizer=None):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            logger.info("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_timeids = np.asarray(self.timeids[:self.num_query]) if cross_time else None
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_timeids = np.asarray(self.timeids[self.num_query:]) if cross_time else None
        if self.reranking:
            logger.info('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            logger.info('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        if cross_time:
            logger.info('=> using cross time evaluation')

        cmc, all_AP, all_INP, query_keep_indices, gallery_keep_indices = eval_func(
            distmat, q_pids, g_pids, q_camids, g_camids, q_timeids, g_timeids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)

        if visualizer is not None:
            logger.info('=> Visualizing results')
            # visualize
            datasets = []
            for i, pid in enumerate(self.pids):
                # , self.camids, self.timeids, self.img_paths
                data = {}
                data['targets'] = pid
                data['camids'] = self.camids[i]
                data['timeids'] = self.timeids[i]
                # data['viewids'] = self.viewids[i]
                data['img_paths'] = self.img_paths[i]
                data['images'] = self.images[i]
                datasets.append(data)
            visualizer.reset(datasets)
            visualizer.get_model_output(
                all_AP, query_keep_indices, gallery_keep_indices,
                distmat, q_pids, g_pids, q_camids, g_camids
            )
            visualizer.plot_img = False
            model_name = self.cfg.MODEL.ARCH_NAME if self.cfg is not None else 'ViT'
            visualizer.vis_rank_list(f'rank_list/{model_name}', max_rank=10)

        return cmc, mAP, mINP, distmat, self.pids, self.camids, qf, gf

    def new_compute(self, time_mode='mix', view_mode='mix', visualizer=None):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            logger.info("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        qf = feats[:self.num_query]  # query
        gf = feats[self.num_query:]  # gallery
        if self.reranking:
            logger.info('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            logger.info('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, all_AP, all_INP, query_keep_indices, gallery_keep_indices = self.eval_func(distmat, time_mode, view_mode)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)

        if visualizer is not None:
            logger.info('=> Visualizing results')
            # visualize
            datasets = []
            for i, pid in enumerate(self.pids):
                # , self.camids, self.timeids, self.img_paths
                data = {}
                data['targets'] = pid
                data['camids'] = self.camids[i]
                data['timeids'] = self.timeids[i]
                # data['viewids'] = self.viewids[i]
                data['img_paths'] = self.img_paths[i]
                data['images'] = self.images[i]
                datasets.append(data)
            visualizer.reset(datasets)
            visualizer.get_model_output_v2(all_AP, query_keep_indices, gallery_keep_indices, distmat, self.pids, self.camids)
            visualizer.plot_img = False
            model_name = self.cfg.MODEL.ARCH_NAME if self.cfg is not None else 'ViT'
            visualizer.vis_rank_list(f'rank_list/{model_name}', max_rank=10)

        return cmc, mAP, mINP, distmat, self.pids, self.camids, qf, gf

    def eval_func(self, distmat, time_mode='mix', view_mode='mix', max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
        """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            logger.info("Note: number of gallery samples is quite small, got {}".format(num_g))
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if len(self.timeids) != 0:
            q_timeids = np.asarray(self.timeids[:self.num_query])
            g_timeids = np.asarray(self.timeids[self.num_query:])
        else:
            q_timeids = None
            g_timeids = None
        if len(self.viewids) != 0:
            q_viewids = np.asarray(self.viewids[:self.num_query])
            g_viewids = np.asarray(self.viewids[self.num_query:])
        else:
            q_viewids = None
            g_viewids = None
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        all_INP = []
        gallery_keep_indices = []  # gallery keep indices [[3, 0, ...], [...], ...]
        query_keep_indices = []

        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            # filter gallery with time mode
            if time_mode == 'mix':
                pass
            elif time_mode == 'cross':
                remove_time = (g_timeids[order] == q_timeids[q_idx])
                remove = remove | remove_time
            elif time_mode == 'same':
                remove_time = (g_timeids[order] != q_timeids[q_idx])
                remove = remove | remove_time
            # filter gallery with view mode
            if view_mode == 'mix':
                pass
            elif view_mode == 'cross':
                remove_view = (g_viewids[order] == q_viewids[q_idx])
                remove = remove | remove_view
            elif view_mode == 'same':
                remove_view = (g_viewids[order] != q_viewids[q_idx])
                remove = remove | remove_view
            # keep gallery samples that are not removed
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            # calculate INP
            pos_idx = np.where(orig_cmc == 1)
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

            query_keep_indices.append(q_idx)
            gallery_keep_indices.append(keep)

        num_valid_q = len(query_keep_indices)
        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        if not self.is_training:
            hard_query = set([q_pids[q_idx] for i, q_idx in enumerate(query_keep_indices) if all_cmc[i][5] == 0])  # rank5 == 0 and remove duplicates
            hard_query = sorted(list(hard_query))
            logger.info(f'Hard queries: {len(hard_query)}')
            valid_qids = list(np.unique(q_pids[query_keep_indices]))
            logger.info(f'num_valid_q: {num_valid_q}; valid queries:({len(valid_qids)})')
        all_cmc = all_cmc.sum(0) / num_valid_q

        return all_cmc, all_AP, all_INP, query_keep_indices, gallery_keep_indices
