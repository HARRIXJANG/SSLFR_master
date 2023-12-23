# SSLFR-master
Created by Hang Zhang from Northwestern Polytechnical University. 

## Introduction
SSLFR is a point cloud Self-Supervised Learning (SSL) framework for machining feature recognition. The framework is pre-trained using a large amount of unlabeled point cloud data, first. Afterward, labeled data is used to fine-tune the pre-trained framework. Finally, the fine-tuned framework can be directly employed for machining feature recognition.

## Setup
(1)	cuda 11.6.112     
(2)	python 3.8.13  
(3)	pytorch 1.12.0   
(4)   tensorboard 2.10.0   

The code is tested on Intel Core i9-10980XE CPU, 128GB memory, and NVIDIA GeForce RTX 3090 GPU. 

## Pretrain
(1)	Get the source code by cloning the repository: https://github.com/HARRIXJANG/SSLFR_master.git.   
(2)	Create a folder named data in the root directory.  
(3)	Download the [dataset](https://drive.google.com/drive/folders/1FWEzZTyYV4E4kksBGu3RGHdx_yT1N1zC?usp=sharing). Point clouds are stored in h5 files. Specifically, point cloud data from ASIN is stored in train.h5 and validation.h5. The point cloud data from MFInstSeg is stored in train_MFInstSeg_pt.h5 and valid_MFInstSeg_pt.h5.  Each point cloud in these datasets has an attribute size of 10, encompassing coordinates, normal vectors, and face types in order. The face type is encoded using a one-hot representation with a length of 4.      
(4)	Run `Pretrain.py` to train the framework.    

## Evaluation
The folder "all_eval_data" contains all public evaluation part graphs.  
(1)	Get the source code by cloning the repository: https://github.com/HARRIXJANG/DRLFS_master.git.   
(2)   Copy a txt file that you want to predict from the "all_eval_data" folder to the "eval" folder (only one file can be placed in the "eval" folder at a time).   
(3)	Run `Evaluation.py` to predict. The result is a txt file, in which each row is the handle numbers of all faces that can contruct an isotlated machining feature (the handle number of a face is unique and constant in NX 12.0). According to the result, readers can manually view the final display result in the corresponding CAD model from NX 12.0.    

## Citation
If you use this code please cite:  
```
@inproceedings{  
      title={A novel method for intersecting machining feature segmentation via deep reinforcement learning},  
      author={Hang Zhang, Wenhu Wang, Shusheng Zhang, Yajun Zhang, Jingtao Zhou, Zhen Wang, Bo Huang, and Rui Huang},  
      booktitle={Advanced Engineering Informatics},  
      year={2023}  
    }
``` 
If you have any questions about the code, please feel free to contact me (zhnwpu714@163.com).
