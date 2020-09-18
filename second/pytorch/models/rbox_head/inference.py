# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rbox_coder import RBoxCoder
from second.pytorch.core import box_torch_ops



class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, score_thresh=0.02, nms=0.5, detections_per_img=100, box_coder=None
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = RBoxCoder(weights=(10., 10., 10, 5,  5., 5., 10.))
        self.box_coder = box_coder

    def forward(self, x, boxes, res_scores, post_max_into_pre_max, num_of_fwd_left=0):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        class_logits, box_regression = x
        #class_prob = F.softmax(class_logits, -1)
        class_prob = torch.sigmoid(class_logits)
        # TODO think about a representation of batch of boxes
        # box_regression
        #print(self.nms)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        ###relative
        proposals = self.box_coder.decode_ori(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        num_classes = class_prob.shape[1]
        proposals2 = self.box_coder.decode2(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) ###do nothing
        proposals3 = self.box_coder.decode3_ori(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) ##do decode on only xy wl angle
        proposals4 = self.box_coder.decode3_ori(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        proposals = proposals.split(boxes_per_image, dim=0)
        proposals2 = proposals2.split(boxes_per_image, dim=0)
        proposals3 = proposals3.split(boxes_per_image, dim=0)
        proposals4 = proposals4.split(boxes_per_image, dim=0)

        class_prob = class_prob.split(boxes_per_image, dim=0)

        #if len(self._post_center_range) > 0:
        post_center_range = torch.tensor([0, -40, -2.2, 70.4, 40, 0.8],
            device=box_regression.device).float()
        predictions_dicts2=[]
        predictions_dicts3=[]
        predictions_dicts=[]
        predictions_dicts_old=[]
        results = []
        for cls_preds, box_preds, box_original, box_bv_preds, box_bv_reg_only, image_shape, res_score  in zip(
            class_prob, proposals, proposals2, proposals3, proposals4, image_shapes, res_scores):

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            box_original= box_original.float()
            total_scores = torch.sigmoid(cls_preds)
            box_bv_preds=box_bv_preds.float()
            box_bv_reg_only=box_bv_reg_only.float()
            # Apply NMS in birdeye view
            nms_func = box_torch_ops.rotate_nms

            #feature_map_size_prod = batch_box_preds.shape[
            #   #                         1] // self.target_assigner.num_anchors_per_location

                # get highest score per prediction, than apply nms
                # to remove overlapped box.
            top_scores = total_scores.squeeze(-1)
            top_labels = torch.zeros(
                total_scores.shape[0],
                device=total_scores.device,
                dtype=torch.long)
            #else:
            #    top_scores, top_labels = torch.max(
            #        total_scores, dim=-1)

            top_scores_old=res_score["scores"]
            top_labels_old = torch.zeros(
                top_scores_old.shape[0],
                device=total_scores.device,
                dtype=torch.long)
            #if self._nms_score_thresholds[0] > 0.0:
            #    top_scores_keep = top_scores >= self._nms_score_thresholds[0]
            #    top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores_old.shape[0] != 0:  ##on
                #if self._nms_score_thresholds[0] > 0.0:
                #    box_preds = box_preds[top_scores_keep]
                #    if self._use_direction_classifier:  ##on
                #        dir_labels = dir_labels[top_scores_keep]
                #    top_labels = top_labels[top_scores_keep]
                boxes_for_nms_old= box_bv_reg_only[:, [0, 1, 3, 4, 6]]

                #if not self._use_rotate_nms:
                #    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                #        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                #        boxes_for_nms[:, 4])
                #    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                #        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected_old = nms_func(  ##on  here sigmoid score to 0~1 normalized scored
                    boxes_for_nms_old,
                    top_scores_old,
                    pre_max_size=post_max_into_pre_max,
                    post_max_size=100,
                    iou_threshold=0.01,
                )
            else:
                selected_old = []


            if top_scores.shape[0] != 0:  ##on
                #if self._nms_score_thresholds[0] > 0.0:
                #    box_preds = box_preds[top_scores_keep]
                #    if self._use_direction_classifier:  ##on
                #        dir_labels = dir_labels[top_scores_keep]
                #    top_labels = top_labels[top_scores_keep]
                boxes_for_nms3 = box_bv_preds[:, [0, 1, 3, 4, 6]]


                boxes_for_nms2 = box_original[:, [0, 1, 3, 4, 6]]

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                #if not self._use_rotate_nms:
                #    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                #        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                #        boxes_for_nms[:, 4])
                #    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                #        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(  ##on  here sigmoid score to 0~1 normalized scored
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=post_max_into_pre_max,
                    post_max_size=100,
                    iou_threshold=0.01,
                )
                # the nms in 3d detection just remove overlap boxes.
                selected2 = nms_func(  ##on  here sigmoid score to 0~1 normalized scored
                    boxes_for_nms2,
                    top_scores,
                    pre_max_size=post_max_into_pre_max,
                    post_max_size=100,
                    iou_threshold=0.01,
                )
                selected3 = nms_func(  ##on  here sigmoid score to 0~1 normalized scored
                    boxes_for_nms3,
                    top_scores,
                    pre_max_size=post_max_into_pre_max,
                    post_max_size=100,
                    iou_threshold=0.01,
                )
            else:
                selected = []
                selected2 = []
                selected3 = []

            # if selected is not None:
            selected_boxes = box_preds[selected]

            #if self._use_direction_classifier:  ##on
            #    selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            selected_boxes2 = box_original[selected2]

            # if self._use_direction_classifier:  ##on
            #    selected_dir_labels = dir_labels[selected]
            selected_labels2 = top_labels[selected2]
            selected_scores2 = top_scores[selected2]


            selected_boxes3 = box_bv_preds[selected3]

            # if self._use_direction_classifier:  ##on
            #    selected_dir_labels = dir_labels[selected]
            selected_labels3 = top_labels[selected3]
            selected_scores3 = top_scores[selected3]
            # finally generate predictions.
            selected_boxes_old = box_bv_reg_only[selected_old]
            selected_labels_old = top_labels_old[selected_old]
            selected_scores_old = top_scores_old[selected_old]

            if selected_boxes_old.shape[0] != 0:
                box_bv_reg_only = selected_boxes_old
                scores_old= selected_scores_old
                label_preds_old = selected_labels_old
                final_box_preds_old = box_bv_reg_only
                final_scores_old = scores_old
                final_labels_old = label_preds_old
                if post_center_range is not None:
                    mask = (final_box_preds_old[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds_old[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict_old = {
                        "box3d_lidar": final_box_preds_old[mask],
                        "scores": final_scores_old[mask],
                        "label_preds": label_preds_old[mask],
                    }
                else:
                    predictions_dict_old = {
                        "box3d_lidar": final_box_preds_old,
                        "scores": final_scores_old,
                        "label_preds": label_preds_old,

                    }
            else:
                dtype = class_prob.dtype
                device = class_prob.device
                predictions_dict_old = {
                    "box3d_lidar":
                        torch.zeros([0, box_bv_reg_only.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts_old.append(predictions_dict_old)




            if selected_boxes3.shape[0] != 0:
                box_bv_preds = selected_boxes3
                scores3 = selected_scores3
                label_preds3 = selected_labels3
                final_box_preds3 = box_bv_preds
                final_scores3 = scores3
                final_labels3 = label_preds3
                if post_center_range is not None:
                    mask = (final_box_preds3[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds3[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict3 = {
                        "box3d_lidar": final_box_preds3[mask],
                        "scores": final_scores3[mask],
                        "label_preds": label_preds3[mask],
                    }
                else:
                    predictions_dict3 = {
                        "box3d_lidar": final_box_preds3,
                        "scores": final_scores3,
                        "label_preds": label_preds3,

                    }
            else:
                dtype = class_prob.dtype
                device = class_prob.device
                predictions_dict3 = {
                    "box3d_lidar":
                        torch.zeros([0, box_bv_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts3.append(predictions_dict3)

            if selected_boxes2.shape[0] != 0:
                box_original = selected_boxes2
                scores2 = selected_scores2
                label_preds2 = selected_labels2
                final_box_preds2 = box_original
                final_scores2 = scores2
                final_labels2 = label_preds2
                if post_center_range is not None:
                    mask = (final_box_preds2[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds2[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict2 = {
                        "box3d_lidar": final_box_preds2[mask],
                        "scores": final_scores2[mask],
                        "label_preds": label_preds2[mask],
                    }
                else:
                    predictions_dict2 = {
                        "box3d_lidar": final_box_preds2,
                        "scores": final_scores2,
                        "label_preds": label_preds2,

                    }
            else:
                dtype = class_prob.dtype
                device = class_prob.device
                predictions_dict2 = {
                    "box3d_lidar":
                        torch.zeros([0, box_original.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts2.append(predictions_dict2)

            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,

                    }
            else:
                dtype = class_prob.dtype
                device = class_prob.device
                predictions_dict = {
                    "box3d_lidar":
                        torch.zeros([0, box_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                    "scores":
                        torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                }
            predictions_dicts.append(predictions_dict)


        return predictions_dicts , predictions_dicts2, predictions_dicts3, predictions_dicts_old


    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 5:(j + 1) * 5]`.
        """
        boxes = boxes.reshape(-1, 7)
        scores = scores.reshape(-1)
        boxlist = RBoxList(boxes, image_shape, mode="xywha")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 7)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        #for j in range(1, num_classes):
        for j in range(0, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 7 : (j + 1) * 7]
            boxlist_for_class = RBoxList(boxes_j, boxlist.size, mode="xywha")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms, score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device) #chchange
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)
        print("number_of_detection")
        print(number_of_detections)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        print("number_of_final_detection")
        print(len(result))
        return result




        #batch_box_preds = preds_dict["box_preds"]
        #batch_cls_preds = preds_dict["cls_preds"]
        #print("box regression is what")
        #print(batch_box_preds.shape)
        #batch_box_preds = batch_box_preds.view(batch_size, -1,
         #                                      self._box_coder.code_size)
        #print(batch_box_preds.shape)
        #num_class_with_bg = self._num_class
        #if not self._encode_background_as_zeros:
        #    num_class_with_bg = self._num_class + 1

        #batch_cls_preds = batch_cls_preds.view(batch_size, -1,
        #                                       num_class_with_bg)
        #print("thisisbatch box_preds")
        #print(batch_box_preds, batch_box_preds.size())
        #print("thisis batch_anchors")
        #print(batch_anchors, batch_anchors.size())
        #batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
        #                                               batch_anchors)

        #if self._use_direction_classifier:
        #    batch_dir_preds = preds_dict["dir_cls_preds"]
        #    batch_dir_preds = batch_dir_preds.view(batch_size, -1,
        #                                           self._num_direction_bins)
        #else:
        #    batch_dir_preds = [None] * batch_size













def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.RBBOX_REG_WEIGHTS
    box_coder = RBoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    postprocessor = PostProcessor(
        score_thresh, nms_thresh, detections_per_img, box_coder
    )
    return postprocessor
