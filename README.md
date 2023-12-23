# SSLFR-master
Created by Hang Zhang from Northwestern Polytechnical University. 

## Introduction
SSLFR is a point cloud Self-Supervised Learning (SSL) framework for machining feature recognition. The framework is pre-trained using a large amount of unlabeled point cloud data, first. Afterward, labeled data is used to fine-tune the pre-trained framework. Finally, the fine-tuned framework can be directly employed for machining feature recognition.

## Setup
(1)	cuda 11.6.112     
(2)	python 3.8.13  
(3)	pytorch 1.12.0   
(4)	tianshou 0.4.11  
(5)   dgl 0.9.1  
(6)   gym 0.25.2  
(7)   tensorboard 2.10.0   

The code is tested on Intel Core i9-10980XE CPU, 128GB memory, and NVIDIA GeForce RTX 3090 GPU. 

## Train
(1)	Get the source code by cloning the repository: https://github.com/HARRIXJANG/DRLFS_master.git.   
(2)	Create a folder named `train_data` and a folder named `test_data` in the folder `Data`.  
(3)	Download the [training dataset](https://drive.google.com/drive/folders/1FWEzZTyYV4E4kksBGu3RGHdx_yT1N1zC?usp=sharing) and the [test dataset](https://drive.google.com/drive/folders/1M-wEQFi1_7Ng03HVYAkw5ynjKU_ptEID?usp=sharing). Graphs are stored in the txt files. Lines starting with #N in the txt indicate the attributes of nodes (see the paper for details), and the last attribute indicates the handle number of a face (for confidentiality reasons, we have hidden the handle numbers of the faces in the training dataset and test dataset). Lines starting with #E represent the attributes of edges, where the first element stands for the source node and the second element for the target node.  
(4)	Put the datasets in the folders `train_data` and `test_data`, respectively.    
(5)	Run `Train.py` to train the framework.    

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
