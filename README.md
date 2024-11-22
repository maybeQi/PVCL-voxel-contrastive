# CVCL
This repository will hold the PyTorch implementation of the  paper [3D medical image segmentation based on prototype-guided voxel contrast learning](). This is an extended implementation for the LA and BraTS benchmark.

## Introduction
![image](https://github.com/maybeQi/Voxel-Contrastive/blob/master/image/%E5%9B%BE%E7%89%871.png)
### Abstract
Semi-supervised learning significantly drives the development of medical image segmentation and alleviating the burden of obtaining high-cost expert annotation. In particular, methods based on contrast learning have much attention due to their excellent performance. Such methods not only explore the semantic information in the local context of a single image, but also capture the global semantic information of small batches or even the whole dataset. However, existing image-level and region-level contrast learning fail to fully consider the processing of fuzzy voxels in target boundaries when refining segmentation. Example-level contrast learning mainly generates different views of the same image through data enhancement to select positive and negative sample pairs, so as to improve the differentiation between categories, but also ignores the global information or context information between samples, that is, the rich semantic relationship between pixels of different images. Moreover, in medical images, strong data augmentation may introduce excessive bias due to its boundary details and sensitivity to construct ideal sample similarity relationships. For the above problems, inspired by the progress in unsupervised contrast representation learning, we propose a new method that focuses on the boundary voxels of segmented targets and mining key voxels around the segmented targets in different samples to construct pixel-wise contrast learning sample pairs for semi-supervised medical image segmentation. Moreover, the current prototype learning method learns a meaningful feature representation through the similarity measurement when data annotation is scarce, enabling the model to perform better in subsequent segmentation tasks. Based on this observation, we propose a novel target voxel screening scheme to select the most valuable difficult positive and difficult negative samples in the different sample segmentation boundaries for contrast learning through the similarity of voxel features to the category prototype. To achieve this goal, we have constructed a new class-level prototype contrast learning (PCL) and voxel-level prototype contrast learning (VCL) framework. This framework screens the difference part of the predicted model by labeling the prototype metric and builds the contrast learning pairs as a negative sample voxel. These two processes work together to encourage feature discriminability and compactness, thus enhancing the performance of the segmentation network. The code is viewed on the https://github.com/maybeQi/Voxel-Contrastive. 

### Highlights
- Utilize prototype-based measure  to achieve "voxel learning" for unlabeled data.

![image](https://github.com/maybeQi/Voxel-Contrastive/blob/master/image/%E5%9B%BE%E7%89%872.png)

## Requirements
Check requirements.txt.
* Pytorch version >=0.4.1.
* Python == 3.6 

## Datesets
Will put them on a public web disk or contact me
## Usage

1. Clone the repo:
```
cd ./CVCL
```

2. Data Preparation
Refer to ./data for details


3. Train
```
cd ./code
python train_CPCL_general_3D.py --labeled_num 8 --model vnet_MTPD --gpu 0 
```

4. Test 
```
cd ./code
python test_3D.py --model vnet_MTPD
```


## Citation

If you find this paper useful, please cite as:
```
@article{xu2022all,
  title={All-around real label supervision: Cyclic prototype consistency learning for semi-supervised medical image segmentation},
  author={Xu, Zhe and Wang, Yixin and Lu, Donghuan and Yu, Lequan and Yan, Jiangpeng and Luo, Jie and Ma, Kai and Zheng, Yefeng and Tong, Raymond Kai-yu},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={26},
  number={7},
  pages={3174--3184},
  year={2022},
  publisher={IEEE}
}
```
## Acknowledgements:
Our code is adapted from UAMT, SASSNet, DTC, CPCL，BHPC and SSL4MIS. Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions：
If any questions, feel free to contact me at 'KomberRisk@gmail.com'
