# Multi-former: Multitask Transformer Framework in Unsupervised Domain Adaptation
<img src = "image\Multi_former.png" width="800px">     
<img src = "image\depth.png" width="800px">

## Codes 
coming soon


## Abstract 
coming soon



## Our results

<img src = "image\result1.png" width="600px">

**Synthia——>Cityscapes**

<img src = "image\result1_img.png" width="600px">

(a):image, (b):GT, (c):Wu, (d):Lu, (e):ours

<img src = "image\result2.png" width="600px">
<img src = "image\result3.png" width="600px">

**Synthia——>Mapillary**

<img src = "image\result3_img.png" width="600px">

(a):image, (b):GT, (c):Wu, (d):Lu, (e):ours

## Create a virtual environment

- python>=3.8.6
- cuda>=10.2
- pytorch==1.8.1
- torchvision==0.9.1
- opencv
- mmcv==1.7.1


## Datasets

* **SYNTHIA**: Please follow the instructions [here](http://synthia-dataset.net/downloads/) to download the images.
  We used the *SYNTHIA-RAND-CITYSCAPES (CVPR16)* split. 
  Download the segmentation labels  [here](https://drive.google.com/file/d/1TA0FR-TRPibhztJI5-OFP4iBNaDDkQFa/view?usp=sharing). 
  The Cityscapes dataset directory should have this basic structure:
  
  "source_density_maps" are generated by "gene_sourceDensityMaps.py" in our project.
  
  ```bash
  <project_root>/data/SYNTHIA                           % SYNTHIA dataset root
  ├── RGB
  ├── parsed_LABELS
  ├── Depth
  └── source_density_maps
  ```
  
* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) 
  to download the images and validation ground-truths. 
  The Cityscapes dataset directory should have this basic structure:
  
  ```bash
  <project_root>/data/Cityscapes                       % Cityscapes dataset root
  ├── leftImg8bit
  │   ├── train
  │   └── val
  └── gtFine
      └── val
  ```
  
* **Mapillary**: Please follow the instructions in [Mapillary](https://www.mapillary.com/dataset) 
  to download the images and validation ground-truths. 
  The Mapillary dataset directory should have this basic structure:
  
  ```bash
  <project_root>/data/Mapillary                        % Mapillary dataset root
  ├── train
  │   └── images
  └── validation
      ├── images
      └── labels
  ```



## Train
Download the weights from here [Swin Transformer for ImageNet Classification]( https://github.com/microsoft/Swin-Transformer ) and export weight = YOUR_WEIGHT_PATH.  and put it in  folder  <project_root>/pretrained_models  for initializing backbone.
```bash
$ cd depth_distribution/main/scripts

# STEP-1: Adversarial learning
$ python train.py --expid=<experiment id>
# If you want to train based on the pre training model, you can execute the command
$ python train.py --expid=<experiment id> --pret_model='<trained model path,it should be noted that the pre-trained model name cannot be changed, and the corresponding adversarial network model should be placed in the same folder.>'
#The results of the training will be saved in the record.txt file

# STEP-2: Self training
$ python self_train.py --expid=<experiment id>  --pret_model='<trained model path>'

# STEP-3: test
$ python test.py --expid=<experiment id>  --pret_model='<trained model path>'
```
Where experiment ids are:
| Exp. Id | Description |
| -------- | -------- |
|  1   | SYNTHIA to Cityscapes 16 classes |
|  2    | SYNTHIA to Cityscapes 7 classes |
|  3   | SYNTHIA to Cityscapes 7 classes (low resolution) |
|  4    | SYNTHIA to Mapillary 7 classes |
|  5   | SYNTHIA to Mapillary 7 classes (low resolution) |

## Test our weights
Download our trained model at [OneDriver](https://1drv.ms/f/s!AnoMEIeojRYtijOkEIlOpwIwkfnq?e=0Ob4jQ)

# Acknowledgements
This codebase depends on [Wu](https://github.com/depdis/Depth_Distribution), [swin_transformer](https://github.com/microsoft/Swin-Transformer), and[Transda](https://github.com/alpc91/TransDA).
Thank you for the work you've done!

