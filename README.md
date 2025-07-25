# [CVPR 2023] Learning Semantic-Aware Disentangled Representation for Flexible 3D Human Body Editing  
Xiaokun Sun, Qiao Feng, Xiongzheng Li, Jinsong Zhang, Yu-Kun Lai, Jingyu Yang, Kun Li. "Learning Semantic-Aware Disentangled Representation for Flexible 3D Human Body Editing". In Proc. CVPR, 2023.  
[Project Page](http://cic.tju.edu.cn/faculty/likun/projects/SemanticHuman) | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/SemanticHuman) | [Video](https://www.youtube.com/watch?v=hnrIv1bnZVw) | [新智元](https://mp.weixin.qq.com/s/6v-3nbzYq2hZCsSacycDRw)

**News**
* `2023/3/31` The code of version 1.0 has been released, welcome to ask me questions!

## Overview
<p align="center">
<img src=img.jpg />
</p>

## Abstract
3D human body representation learning has received increasing attention in recent years. However, existing works cannot flexibly, controllably and accurately represent human bodies, limited by coarse semantics and unsatisfactory representation capability, particularly in the absence of supervised data. In this paper, we propose a human body representation with fine-grained semantics and high reconstruction-accuracy in an unsupervised setting. Specifically, we establish a correspondence between latent vectors and geometric measures of body parts by designing a part-aware skeleton-separated decoupling strategy, which facilitates controllable editing of human bodies by modifying the corresponding latent codes. With the help of a bone-guided auto-encoder and an orientation-adaptive weighting strategy, our representation can be trained in an unsupervised manner. With the geometrically meaningful latent space, it can be applied to a wide range of applications, from human body editing to latent code interpolation and shape style transfer. Experimental results on public datasets demonstrate the accurate reconstruction and flexible editing abilities of the proposed method. The code will be released for research purposes.

## Dependencies

To run this code, the following packages need to be installed.
My package version is for reference only.
```
python (3.7.13)
numpy (1.21.6)
pytorch (1.10.0)
tensorboardX (2.10.0)
sklearn (1.0.2)
scipy (1.7.3)
[pytorch3d](https://github.com/facebookresearch/pytorch3d) (0.7.0 need to be installed manually)
[psbody](https://github.com/MPI-IS/mesh) (0.4 need to be installed manually)
trimesh (3.14.0)
tqdm (4.64.0)
yacs (0.1.8)
[torch-geometric](https://github.com/MPI-IS/mesh) (2.1.0 need to be installed manually)
torch-cluster (1.5.9)
torch_scatter (2.0.9)
torch_sparse (0.6.12)
torch_spline_conv (1.2.1)
```

## Assets Files

You need to download the assets files, unzip them and put them into the `root/asset` directory.

[Download: assets](https://pan.baidu.com/s/1IDPlUgyAPRkfMVVt_w2R8Q?pwd=dxvl)

## Preprocessed Data and Pretrain Model

You need to download the preprocessed data and pretrain model, unzip them and put them into the `root/DFAUST` directory.

[Download: preprocessed DFAUST dataset and pretrained model](https://pan.baidu.com/s/1uRjvLSCtWLwr6AZbhbqh5w?pwd=5fa9)

## Usage

### Configure
Update `cfg.PATH.root_dir` in the `root/configure/cfgs.py` file.  
Update `cfg.TRAIN.resume` in the `root/configure/testcfg.yaml` file.

### Data preprocessing 
The downloaded preprocessed data can be trained directly.  
If you want to train on your own dataset, please use the following code. (Measurement parameters created by data_generation.py)
```
$ python obj2npy.py --save_path data_path --trainobj_path obj_path1 --testobj_path obj_path2 --train_start 0 --train_end 1000 --test_start 0 --test_end 100  
$ python data_generation.py -r root_path -d dataset_name --train_measure mea_path1 --test_measure mea_path2 -v 0
```

### Training
Run the following code to train the network.
```
$ python main.py
```

### Editing

Run the following code to output editing results.
```
$ python demo.py
```

## Acknowledgement
Part of our code is based on [Neural3DMM](https://github.com/gbouritsas/Neural3DMM). Many thanks! 

## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```
@inproceedings{SemanticHuman,
  author = {Xiaokun Sun, Qiao Feng, Xiongzheng Li, Jinsong Zhang, Yu-Kun Lai, Jingyu Yang, Kun Li},
  title = {Learning Semantic-Aware Disentangled Representation for Flexible 3D Human Body Editing},
  booktitle = {CVPR},
  year={2023},
}
```
