# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class RBoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 3]# - proposals[:, 0] + TO_REMOVE
        ex_lengths = proposals[:, 4]# - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0]# + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1]# + 0.5 * ex_heights
        ex_ctr_z = proposals[:, 2]
        ex_angle = proposals[:, 6]
        ex_re_heights=proposals[:, 5]

        gt_widths = reference_boxes[:, 3]# - reference_boxes[:, 0] + TO_REMOVE
        gt_lengths = reference_boxes[:, 4]# - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0]# + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1]# + 0.5 * gt_heights
        gt_ctr_z = reference_boxes[:, 2]
        gt_angle = reference_boxes[:, 6]
        gt_re_heights=reference_boxes[:,5]

        diagonal=torch.sqrt(ex_widths**2+ex_lengths**2)
        wx, wy, wz, ww, wl, wh, wa = self.weights


        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / diagonal
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / diagonal
        targets_dz = wz * (gt_ctr_z - ex_ctr_z) / ex_re_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dl = wl * torch.log(gt_lengths / ex_lengths)
        targets_dh = wh * torch.log(gt_re_heights / ex_re_heights)
        targets_da = wa * (gt_angle - ex_angle)
        #targets_da[np.where((gt_angle <= -30) & (ex_angle >= 120))] += 180
        #targets_da[np.where((gt_angle >= 120) & (ex_angle <= -30))] -= 180

        #gtle30 = gt_angle.le(-30)
        #exge120 = ex_angle.ge(120)
        #gtge120 = gt_angle.ge(120)
        #exle30 = ex_angle.le(-30)

        #incre180 = gtle30 * exge120 * 180
        #decre180 = gtge120 * exle30 * (-180)

        #targets_da = targets_da + incre180.float()
        #targets_da = targets_da + decre180.float()

        #targets_da = 3.14159265358979323846264338327950288 / 180 * targets_da

        targets = torch.stack((targets_dx, targets_dy, targets_dz,targets_dw, targets_dl,targets_dh, targets_da), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 3]# - boxes[:, 0] + TO_REMOVE
        lengths = boxes[:, 4]# - boxes[:, 1] + TO_REMOVE
        heights = boxes[:, 5]# - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0]# + 0.5 * widths
        ctr_y = boxes[:, 1]# + 0.5 * heights
        ctr_z = boxes[:, 2]# + 0.5 * heights

        angle = boxes[:, 6]

        wx, wy, wz, ww, wl, wh, wa = self.weights
        dx = rel_codes[:, 0::7] / wx
        dy = rel_codes[:, 1::7] / wy
        dz = rel_codes[:, 2::7] / wz
        dw = rel_codes[:, 3::7] / ww
        dl = rel_codes[:, 4::7] / wl
        dh = rel_codes[:, 5::7] / wh
        da = rel_codes[:, 6::7] / wa
        #diagonal=torch.sqrt(ex_widths**2+ex_heights**2)
        diagonal= torch.sqrt(lengths**2+widths**2)
        pred_ctr_x = dx * diagonal + ctr_x[:, None]
        pred_ctr_y = dy * diagonal + ctr_y[:, None]
        pred_ctr_z = dz * heights + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        #da = da * 1.0 / 3.141592653 * 180  # arc to angle
        pred_angle = da + angle[:, None]

        # print('pred_angle:', pred_angle.size())
        pred_boxes = torch.zeros_like(rel_codes)
        # ctr_x1
        pred_boxes[:, 0::7] = pred_ctr_x
        # ctr_y1
        pred_boxes[:, 1::7] = pred_ctr_y

        pred_boxes[:, 2::7] = pred_ctr_z
        # width
        pred_boxes[:, 3::7] = pred_w
        # height
        pred_boxes[:, 4::7] = pred_l

        pred_boxes[:, 5::7] = pred_h
        # angle
        pred_boxes[:, 6::7] = pred_angle

        return pred_boxes