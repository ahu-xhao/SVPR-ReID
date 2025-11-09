# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from cv2 import add
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


class ReIDLoss:
    def __init__(self, cfg, num_classes, feat_dim=2048):
        self.cfg = cfg
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.sampler = cfg.DATALOADER.SAMPLER
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes) if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.NO_MARGIN:
                self.triplet = TripletLoss()
                print("using soft triplet loss for training")
            else:
                self.triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
                print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
        else:
            print('expected METRIC_LOSS_TYPE should be triplet'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    def __call__(self, score, feat, target, target_cam=None, i2tscore=None, add_type='sum'):
        loss_id = self.get_id_loss(score, feat, target, add_type=add_type)
        loss_tri = self.get_triplet_loss(score, feat, target, add_type=add_type)
        loss_reid = loss_id + loss_tri
        if i2tscore is not None:
            loss_i2t = self.i2t_loss(i2tscore, target)
            loss_reid += loss_i2t

        return loss_reid

    def get_id_loss(self, score, feat, target, add_type='sum'):
        id_loss_weight = self.cfg.MODEL.ID_LOSS_WEIGHT
        if self.sampler == 'softmax':
            ID_LOSS = F.cross_entropy(score, target)
            return ID_LOSS

        if self.cfg.MODEL.METRIC_LOSS_TYPE != 'triplet':
            print('expected METRIC_LOSS_TYPE should be triplet'
                  'but got {}'.format(self.cfg.MODEL.METRIC_LOSS_TYPE))
            return 0.

        if not isinstance(score, list):
            score = [score]
        if add_type == 'sum':
            id_loss = [self.xent(score_i, target) for score_i in score]
            ID_LOSS = sum(id_loss)
        elif add_type == 'mean':
            id_loss = [self.xent(score_i, target) for score_i in score]
            ID_LOSS = sum(id_loss) / len(id_loss) if len(id_loss) > 0 else 0.
        else:
            id_loss_0 = self.xent(score[0], target)
            id_loss_1 = [self.xent(scor, target) for scor in score[1:]]
            id_loss_1 = sum(id_loss_1) / len(id_loss_1) if len(id_loss_1) > 0 else 0.
            ID_LOSS = id_loss_0 + id_loss_1
        id_loss = ID_LOSS * id_loss_weight
        return id_loss

    def get_triplet_loss(self, score, feat, target, add_type='sum'):
        triplet_loss_weight = self.cfg.MODEL.TRIPLET_LOSS_WEIGHT
        if not isinstance(feat, list):
            feat = [feat]
        if add_type == 'sum':
            tri_loss = [self.triplet(feat_i, target)[0] for feat_i in feat]
            TRI_LOSS = sum(tri_loss)
        elif add_type == 'mean':
            tri_loss = [self.triplet(feat_i, target)[0] for feat_i in feat]
            TRI_LOSS = sum(tri_loss) / len(tri_loss) if len(tri_loss) > 0 else 0.
        else:
            tri_loss_0 = self.triplet(feat[0], target)[0]
            tri_loss_1 = [self.triplet(feats, target)[0] for feats in feat[1:]]
            tri_loss_1 = sum(tri_loss_1) / len(tri_loss_1) if len(tri_loss_1) > 0 else 0.
            TRI_LOSS = tri_loss_0 + tri_loss_1

        triplet_loss = triplet_loss_weight * TRI_LOSS
        return triplet_loss

    def i2t_loss(self, i2tscore, targets=None):
        if targets is None:
            # unique_pids, targets = torch.unique(targets, return_inverse=True)
            targets = torch.arange(i2tscore.shape[0]).to(i2tscore.device)

        # assert targets.max() < i2tscore.shape[0], "label should be less than batch size"
        i2t_loss_weight = self.cfg.MODEL.I2T_LOSS_WEIGHT
        I2TLOSS = self.xent(i2tscore, targets)
        I2TLOSS = I2TLOSS * i2t_loss_weight

        return I2TLOSS

    def get_view_loss(self, score, feat_view, feat_cls, view, view_lambda=1.):
        if self.sampler == 'softmax':
            VIEW_ID_LOSS = F.cross_entropy(score, view)
            return VIEW_ID_LOSS

        VIEW_ID_LOSS = self.xent(score, view)
        view_id_loss = VIEW_ID_LOSS * view_lambda
        view_orthogonal = torch.cosine_similarity(feat_cls, feat_view).abs().mean() * view_lambda
        view_loss = view_id_loss + view_orthogonal
        return view_loss

    # def get_pts_loss(self, feat_list, margin=0.1):
    #     pts_loss = [torch.cosine_similarity(x, y).abs().mean()
    #                 for i, x in enumerate(feat_list) for y in feat_list[i + 1:]]
    #     pts_loss = sum(pts_loss) / len(pts_loss) if len(pts_loss) > 0 else 0.

    #     return pts_loss
    
    def get_pts_loss(self, feat_list, margin=0.1):
        losses = []
        for i in range(len(feat_list)):
            for j in range(i + 1, len(feat_list)):
                cos_sim = F.cosine_similarity(feat_list[i], feat_list[j], dim=-1)
                loss = (1.0 - cos_sim).clamp(min=margin).mean()
                losses.append(loss)
        return sum(losses) / len(losses) if len(losses) > 0 else torch.tensor(0.0, device=feat_list[0].device)