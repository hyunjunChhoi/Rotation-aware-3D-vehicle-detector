# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.visualize import vis_image
from PIL import Image
from second.pytorch.core.losses import Loss,_sigmoid_cross_entropy_with_logits
from second.core import box_np_ops

_DEBUG = False

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, edge_punished=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.edge_punished = edge_punished

    def match_targets_to_proposals(self, proposal, target):
        box1, box2 = proposal.bbox, target.bbox

        box1_np = box1.data.cpu().numpy()
        box2_np = box2.data.cpu().numpy()
        ch_box1 = box1_np.copy()
        ch_box11 = ch_box1[:, [0, 1, 3, 4, 6]]
        ch_box2 = box2_np.copy()
        ch_box22 = ch_box2[:, [0, 1, 3, 4, 6]]

        match_quality_matrix2 = box_np_ops.riou_cc(ch_box22, ch_box11)
        match_quality_matrix2 = torch.from_numpy(match_quality_matrix2).float().to(box1.device)
        #match_quality_matrix = boxlist_iou(proposal, target)

        #print(match_quality_matrix)
        #print(match_quality_matrix2)
        matched_idxs = self.proposal_matcher(match_quality_matrix2)
        # print('matched_idxs', matched_idxs)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        #print("this is target")
        #print(len(target) , target)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, batch_idx):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image, batch_id in zip(proposals, targets, batch_idx):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets

            ##################################
            #print("this is matched target box")
            #print(matched_targets.bbox)
            #print("this is proposal per image box")
            #print(proposals_per_image.bbox)
            ###################################

            regression_targets_per_image = self.box_coder.encode_ori_rel(
                matched_targets.bbox, proposals_per_image.bbox
            )
            #relative
            if _DEBUG:
                label_np = labels_per_image.data.cpu().numpy()
                # print('label shape:', label_np.shape)
                # print('labels pos/neg:', len(np.where(label_np == 1)[0]), '/', len(np.where(label_np == 0)[0]))
                imh, imw = proposals_per_image.size
                proposals_np = proposals_per_image.bbox.data.cpu().numpy()
                canvas = np.zeros((imh, imw, 3), np.uint8)

                # pick pos proposals for visualization
                pos_proposals = proposals_np[label_np == 1]
                print(pos_proposals)
                #pos_proposals = proposals_np[label_np == 1]
                # print('proposals_np:', pos_proposals)
                pos_proposals[:, 0] = (pos_proposals[:, 0]) * 175 / 70.4 + 0.5
                pos_proposals[:, 1] = (pos_proposals[:, 1] + 40.0) * 199 / 80.0 + 0.5
                pos_proposals[:, 3] = (pos_proposals[:, 3]) * 175 / 70.4
                pos_proposals[:, 4] = (pos_proposals[:, 4]) * 199 / 80.0

                pilcanvas = vis_image(Image.fromarray(canvas), pos_proposals[:,[0,1,3,4,6]], [i for i in range(pos_proposals.shape[0])])
                print(batch_id)
                pilcanvas.show()
                pilcanvas.save('proposalmaskboxes.jpg', 'jpeg')


            # print('proposal target', regression_targets_per_image, np.unique(labels_per_image.data.cpu().numpy()))
            # print('labels_per_image:', labels_per_image.size(), np.unique(label_np))
            #print("label per image")
            #print(labels_per_image.size(), labels_per_image)
            #print("regression target per image")
            #print(regression_targets_per_image.size(), regression_targets_per_image)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets, batch_idx):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        # print('targets:', targets[0].bbox)
        labels, regression_targets = self.prepare_targets(proposals, targets, batch_idx)
        # print('regression_targets:', targets[0].bbox)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression, cc_loss, ll_loss, batch_size):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)# .detach()
        box_regression = cat(box_regression, dim=0)#.detach()
        device = class_logits.device

        # print('class_logits:', class_logits.size())

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        if _DEBUG:
            #print('labels:', labels)
            #print('rrpn_labels:', np.unique(labels.data.cpu().numpy()))
            pass
        ##########################################################################
        #print('loss_box_regression:', box_regression, box_regression.size())
        #print('loss_class_logits:', class_logits, class_logits.size())
        #print('labels:', labels, labels.size())
        ##################################################################333
        #if labels.size()
        new_loss=_sigmoid_cross_entropy_with_logits(class_logits, torch.unsqueeze(labels, 1))

        #print(new_loss.size())
        #classification_loss = F.cross_entropy(class_logits, labels)
        #print(labels.size())
        #print(new_loss.size()[0])
        #print(new_loss)
        #print(new_loss.sum())
        cls_weights, reg_weights, cared= prepare_loss_weights(labels.unsqueeze(0),
                             pos_cls_weight=1.0,
                             neg_cls_weight=1.0,
                             dtype=torch.float32)
        #classification_loss=new_loss.sum()/new_loss.size()[0]
        #loc_losses = loc_loss_ftor(
        #    box_preds, reg_targets, weights=reg_weights)  # [N, M]
        #cls_losses = cls_loss_ftor(
        #    cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_targets = labels.unsqueeze(0) * cared.type_as(labels)

        cls_targets_new = cls_targets.float()
        #print(cls_targets_new.size())
        new_new_loc_loss=ll_loss(box_regression.unsqueeze(0),regression_targets.unsqueeze(0), weights=reg_weights)
        new_new_cls_loss=cc_loss(class_logits.unsqueeze(0), cls_targets_new.unsqueeze(-1), weights=cls_weights)
        #print("error here?")
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        # 여기서 뭔가 이상한데 갑자기 로스까 작게나옴 왜이럴까 ?
        #print("this is labels")
        #print(labels, labels.size())
        loc_loss_reduced = new_new_loc_loss.sum()
        #loc_loss_reduced *= self._loc_loss_weight
        #cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        #cls_pos_loss /= self._pos_cls_weight
        #cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = new_new_cls_loss.sum()
        #cls_loss_reduced *= self._cls_loss_weight
        #loss = loc_loss_reduced + cls_loss_reduced



        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        #print("error here? ? >>")
        #print(sampled_pos_inds_subset, sampled_pos_inds_subset.size())

        labels_pos = labels[sampled_pos_inds_subset]
        #print("error here ? ? ??????>>>>>>>")


        ############################################################
        #print("this is labels_pos")
        #print(labels_pos, labels_pos.size())
        ##################################
        #print(labels_pos)
        # pick the target of correct position
        map_inds = 7 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3, 4, 5 ,6], device=device)

        #box_regression_pos = box_regression[sampled_pos_inds_subset[:, None], map_inds] #d여기서 자꾸 에러남
        box_regression_pos = box_regression[sampled_pos_inds_subset[:, None], :]
        regression_targets_pos = regression_targets[sampled_pos_inds_subset]
        #print(self.edge_punished)
        #########################################################################3
        #print("this is box_regression_pos")
        #print(box_regression_pos, box_regression_pos.size())
        #print(regression_targets, regression_targets.size())
        #print("this is regression_target_pos")
        #print(regression_targets_pos, regression_targets_pos.size())
        ###################################################
        if self.edge_punished:
            proposals_cat = torch.cat([proposal.bbox for proposal in proposals], 0)
            proposals_cat_w = proposals_cat[:, 3:4][sampled_pos_inds_subset]
            proposals_cat_w_norm = proposals_cat_w / (torch.mean(proposals_cat_w) + 1e-10)
            box_regression_pos = proposals_cat_w_norm * box_regression_pos
            regression_targets_pos = proposals_cat_w_norm * regression_targets_pos

        if _DEBUG:
            # print('map_inds:', box_regression[sampled_pos_inds_subset[:, None], map_inds], regression_targets[sampled_pos_inds_subset])
            pass
        box_loss = smooth_l1_loss(
            box_regression_pos,
            regression_targets_pos,
            size_average=False,
            beta=1,
        )

        #box_loss = box_loss / labels.numel() 전체개수로 나눌게아니라 포지티브샘플 개수에 대해서만 나눔
        #
        box_loss = box_loss / labels_pos.numel()
        #############################################
        #print(box_loss)
        #print(classification_loss)
        ###############################3
        return cls_loss_reduced, loc_loss_reduced
def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)

     # for focal loss
    pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
    reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)

    return cls_weights, reg_weights, cared

def make_roi_box_loss_evaluator(cfg):
    #print(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD)
    #print(cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD)

    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.RBBOX_REG_WEIGHTS
    #print(bbox_reg_weights)
    box_coder = RBoxCoder(weights=bbox_reg_weights)
    #print(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    edge_punished = cfg.MODEL.EDGE_PUNISHED
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder, edge_punished)

    return loss_evaluator
