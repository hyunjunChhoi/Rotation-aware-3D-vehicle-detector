# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from second.pytorch.models.rbox_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from second.pytorch.models.rbox_head.roi_box_predictors import make_roi_box_predictor
from second.pytorch.models.rbox_head.inference import make_roi_box_post_processor
from second.pytorch.models.rbox_head.loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.structures import bounding_box

import time

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, additional_info):
        super(ROIBoxHead, self).__init__()
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = additional_info["NUM_CLASSES"]
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD =additional_info["FG_IOU_THRESHOLD"]
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD =additional_info["BG_IOU_THRESHOLD"]
        cfg.MODEL.ROI_HEADS.NMS = additional_info["NMS"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH = additional_info["SCORE_THRESH"]
        cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = additional_info["POOLER_SCALE"]
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = additional_info["POOLER_RESOLUTION"]
        cfg.MODEL.BACKBONE.OUT_CHANNELS = additional_info["OUT_CHANNELS"]
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.cfg = cfg

    def forward(self, feature_final, res, res_score, example=None, batch_idx=None, cc_loss=None, ll_loss=None, post_max_sizes=None,  is_training=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # if self.cfg.TEST.CASCADE:
        recur_iter =1
        targets=example
        features=feature_final
        #print(features[0].type)
        recur_proposals = res
        x = None
        for i in range(recur_iter):

            if is_training:
                # Faster R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    recur_proposals = self.loss_evaluator.subsample(recur_proposals, targets,  batch_idx)

            #print("this is proposal")
            #print(len(recur_proposals))
            #print(recur_proposals[0].bbox[0], recur_proposals[0].bbox[1])
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            #recur_proposals_before_feature
            #print(features[0].size())
            #print("thisis lenght of fatures")
            #print(len(features))
            torch.cuda.synchronize()
            t0=time.time()
            x = self.feature_extractor(features, recur_proposals)
            #print(x.size())
            #print(x.type)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x)
            #print("1")
            #print(class_logits.size())
            #print(box_regression.size())
            torch.cuda.synchronize()
            inference_time=time.time()-t0
            print("{} seconds".format(time.time()-t0))
            if not is_training:
                #print("2")
                recur_proposals, recur2, recur3, recur4 = self.post_processor((class_logits, box_regression), recur_proposals, res_score, post_max_sizes, recur_iter - i - 1) # result
            else:
                #print("3")
                loss_classifier, loss_box_reg = self.loss_evaluator(
                    [class_logits], [box_regression], cc_loss, ll_loss, len(features)
                )
                #recur_proposals = self.post_processor((class_logits, box_regression), recur_proposals, recur_iter - i - 1) # result
        if not is_training:
            #print("4")
            return inference_time, recur_proposals, recur2, recur3, recur4

        return (
            class_logits,
            recur_proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, additional_info):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, additional_info)
