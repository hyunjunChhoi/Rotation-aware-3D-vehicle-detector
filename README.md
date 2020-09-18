
This repo demonstrates how to reproduce the results "Rotation-aware-3D-vehicle-detector"
This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch)
We make minimum changes for implementing our algorithms (Using SECOND as backbone network) 
Evaluation is on the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/)


## Getting Started

This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch) and the relevant
subset of the original README is reproduced here.

## Installation 
For environment settings, we follow this repo; [PointPillars](https://github.com/SmallMunich/nutonomy_pointpillars/)


### 1. Clone code

```bash
git clone https://github.com/hyunjunChhoi/Rotation-aware-3D-vehicle-detector/git
```

### 2. Install dependence python packages

Following [PointPillars](https://github.com/SmallMunich/nutonomy_pointpillars/)

### 3. Some additional installation (RRPN Module)

We also refers to the open source 

[RRPN]https://github.com/mjq11302010044/RRPN_pytorch 

Follow the installation process 

Install RRPN module 

and then replace it with  ./RRPN_module_modified(Some additional modification) 

### 4. add second.pytorch/ to PYTHONPATH


## Prepare dataset (following SECOND)

* KITTI Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Then run
```bash
python create_data.py kitti_data_prep --data_path=KITTI_DATASET_ROOT
```
## Minor changes from the original code 

- /second/pytorch/models/voxelnet -> /second/pytorch/models/voxeltwonet

- some additional files in /second/pytorch/models (ROI head network)

- Import RRPN module (RROI pooling method)

- some modification on train.py and inference.py and config files


## Usage

### train

1. load pretrained network (original SECOND as backbone, /second/old_voxelnet-61900.tckpt )

2. freeze it and train only ROI_head network 

#### train with single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train.py train --config_path=dir/second/configs/car_two_norotate_ori_no1.fhd.config --model_dir=dir/second/output_voxeltwonet_old_61900_freeze --pretrained_path=dir/second/old_voxelnet-61900.tckpt --pretrained_exclude=ROI_head  --freeze_exclude=ROI_head’
```
### evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python train.py evaluate --config_path=dir/second/configs/car_two_norotate_ori_no1.fhd.config --model_dir=dir/second/output_voxeltwonet_old_61900_freeze
```

## Various experiments 

modify config files in dir/second/configs/



