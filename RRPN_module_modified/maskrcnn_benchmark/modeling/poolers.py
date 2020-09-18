# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.layers import RROIAlign

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class PyramidRROIAlign(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PyramidRROIAlign, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                RROIAlign(
                    output_size, spatial_scale=scale
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox[:,[0,1,3,4,6]] for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois
    def convert_to_roi_list_format(self, boxes):
        roi_list=[]
        for member in boxes:
            #member.bbox[:,1]=torch.add(member.bbox[:,1],40.0)
            #print("this is bounding box")
            #print(member.bbox)
            roi_per_list=member.bbox[:,[0,1,3,4,6]]
            
            roi_per_list[:,0]=(roi_per_list[:,0])*175/70.4+0.5
            roi_per_list[:,1]=(roi_per_list[:,1]+40.0)*199/80.0+0.5
            roi_per_list[:,2]=(roi_per_list[:,2])*175/70.4
            roi_per_list[:,3]=(roi_per_list[:,3])*199/80.0
            """
            #######
            roi_per_list[:,0]=(roi_per_list[:,0])*215/69.12+0.5
            roi_per_list[:,1]=(roi_per_list[:,1]+39.68)*247/79.36+0.5
            roi_per_list[:,2]=(roi_per_list[:,2])*215/69.12
            roi_per_list[:,3]=(roi_per_list[:,3])*247/79.36
             """    
            ids=torch.full((len(member),1),0, dtype=member.bbox.dtype, device=member.bbox.device)
            roi_temp=torch.cat([ids, roi_per_list],dim=1)
            #print(ids.size())
            #print(roi_per_list.size())
            roi_list.append(roi_temp)


        return roi_list

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        #print("num_levels")
        #print(num_levels)
        rois = self.convert_to_roi_format(boxes)
        #print("rois")
        #print(rois)
        roi_list=self.convert_to_roi_list_format(boxes)
        #print("roi_list")
        #print(roi_list)
        #print(len(roi_list))
        #if num_levels == 1:
        #    return self.poolers[0](x, rois)

        #levels = self.map_levels(boxes)

        num_rois = len(rois)
        #print("num rois")
        #print(num_rois)
        #num_channels = x[0].shape[1]
        #output_size = self.output_size[0]
        #print("levels")
        #print(levels)
        #dtype, device = x[0].dtype, x[0].device
        #result = torch.zeros(
        #    (num_rois, num_channels, output_size, output_size),
        #    dtype=dtype,
        #    device=device,
        #)

        result = []

        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            #idx_in_level = torch.nonzero(rois[:,0] == level).squeeze(1)
            #rois_per_level = rois[idx_in_level]
            #result[level] = pooler(per_level_feature, rois_per_level)  #  rois_per_level)
            #print(level)
            #print("feature")
            #print(per_level_feature.unsqueeze(0).size())
            #print("roi_list")
            #print(roi_list[level].size(), roi_list[level])
            result.append(pooler(per_level_feature, roi_list[level]))
            #print(result)
        #for member in result:
        #print(len(result))
        final_result=torch.cat(result, dim=0)
        #print(final_result.size())
        #for i in range(len(result)-1):
        #    final_result=torch.cat([final_result,result[i+1]],dim=0)

        #print(final_result.size())
        return final_result # torch.cat(result, 1)



class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result
