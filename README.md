# CVCL
This repository will hold the PyTorch implementation of the  paper [In order to prevent this part in the check, we temporarily do hiding](). This is an extended implementation for the LA and BraTS benchmark.

## Introduction
![image](https://github.com/maybeQi/Voxel-Contrastive/blob/master/image/780e00fd27851e6580ddf510855e2ab.png)
### Abstract
In order to prevent this part in the check, we temporarily do hiding...
![image](https://github.com/maybeQi/Voxel-Contrastive/blob/master/image/图片1.png)

### Highlights
- Utilize prototype-based measure  to achieve "voxel learning" for unlabeled data.

![image](https://github.com/maybeQi/Voxel-Contrastive/blob/master/image/图片2.png)

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
