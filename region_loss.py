import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale,
                  sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = int(len(anchors) / num_anchors)
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            cur_gt_boxes = torch.tensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # conf_mask[b][cur_ious > sil_thresh] = 0
        temp_thresh = cur_ious > sil_thresh
        conf_mask[b][temp_thresh.view(conf_mask[b].shape)] = 0
    if seen < 12800:
        if anchor_step == 4:
            tx = torch.tensor(anchors).view(nA, anchor_step)[:, 2].view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
            ty = torch.tensor(anchors).view(num_anchors, anchor_step)[:, 2].view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        # 针对每个GT
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            # cell所在的位置
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = [0, 0, gw, gh]
            # 与GT的IOU最大的anchor负责预测该GT
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n])
            th[b][best_n][gj][gi] = math.log(gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t * 5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = int(len(anchors) / num_anchors)
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        nB = output.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.size(2)
        nW = output.size(3)

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = F.sigmoid(output[:, :, 0, :, :].view(nB, nA, nH, nW).contiguous())
        y = F.sigmoid(output[:, :, 1, :, :].view(nB, nA, nH, nW).contiguous())
        w = output[:, :, 2, :, :].view(nB, nA, nH, nW).contiguous()
        h = output[:, :, 3, :, :].view(nB, nA, nH, nW).contiguous()
        conf = F.sigmoid(output[:, :, 4, :, :].view(nB, nA, nH, nW).contiguous())
        cls = output[:, :, 5:5 + nC, :, :].view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        with torch.no_grad():
            pred_boxes = output.new_zeros((4, nB * nA * nH * nW))
            grid_x = output.new_tensor(torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW))
            grid_y = output.new_tensor(torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW))
            anchor_w = output.new_tensor(torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])))
            anchor_h = output.new_tensor(torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])))
            anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
            anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
            pred_boxes[0] = x.detach().view(grid_x.size()) + grid_x
            pred_boxes[1] = y.detach().view(grid_y.size()) + grid_y
            pred_boxes[2] = torch.exp(w.detach().view(anchor_w.size())) * anchor_w
            pred_boxes[3] = torch.exp(h.detach().view(anchor_h.size())) * anchor_h
            pred_boxes = pred_boxes.transpose(0, 1).contiguous().view(-1, 4).cpu()

            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                        target,
                                                                                                        self.anchors, nA, nC, nH, nW,
                                                                                                        self.noobject_scale,
                                                                                                        self.object_scale,
                                                                                                        self.thresh,
                                                                                                        self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().item())

        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        tconf = tconf.cuda()
        tcls = tcls.view(-1)[cls_mask.view(-1)].long().cuda()

        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.sqrt().cuda()
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).cuda()
        cls = cls[cls_mask].view(-1, nC)

        loss_x = self.coord_scale * F.mse_loss(x * coord_mask, tx * coord_mask, reduction='sum') / 2.0
        loss_y = self.coord_scale * F.mse_loss(y * coord_mask, ty * coord_mask, reduction='sum') / 2.0
        loss_w = self.coord_scale * F.mse_loss(w * coord_mask, tw * coord_mask, reduction='sum') / 2.0
        loss_h = self.coord_scale * F.mse_loss(h * coord_mask, th * coord_mask, reduction='sum') / 2.0
        loss_conf = F.mse_loss(conf * conf_mask, tconf * conf_mask, reduction='sum') / 2.0
        loss_cls = self.class_scale * F.cross_entropy(cls, tcls, reduction='sum')
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
            self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
            loss_conf.item(), loss_cls.item(), loss.item()))
        return loss
