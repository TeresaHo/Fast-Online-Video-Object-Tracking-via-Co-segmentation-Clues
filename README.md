### Project: Fast-Online-Video-Object-Tracking-via-Co-segmentation-Clues

### Project description
S2iamMask(Spatial-aware SiamMask)integrates a semantic segmentation branch which can further improve the robustness of the object tracker. Such integration has been done before;however, previous methods use mask branch in a way which result in heavily relying on the output score and thus fail to preserve more spatial information which is crucial for mask generation. To better address this problem, we proposal a two-stage training strategy and directly use the proposals obtained in first stage as the input for second stage. This way, spatial information of feature maps are better reserved. Secondly, we use fusion layers for FCN  to  combine semantic  informationfrom a deeper, coarse layer with appearance information from a shallow, fine layer to produce detailed segmentations. Finally, during inference, we design a voting mechanism to prevent the mask from being heavily dependent on the output score.


<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask.jpg" width="600px" />
</div>


## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing-models)
4. [Training Models](#training-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, RTX 2080 GPUs

- Clone the repository 
```
git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
export SiamMask=$PWD
```
- Setup python environment
```
conda create -n vot python=3.6
source activate vot
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Testing
- [Setup](#environment-setup) your environment
- Download test data
```shell
cd $SiamMask/data
sudo apt-get install jq
bash get_test_data.sh
```
- Download pretrained models
```shell
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Evaluate performance on [VOT](http://www.votchallenge.net/)
```shell
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2016 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2018 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2019 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2016 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2018 0
python ../../tools/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016
python ../../tools/eval.py --dataset VOT2018 --tracker_prefix C --result_dir ./test/VOT2018
python ../../tools/eval.py --dataset VOT2019 --tracker_prefix C --result_dir ./test/VOT2019
```
- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2016 0
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth DAVIS2017 0
```
- Evaluate performance on [Youtube-VOS](https://youtube-vos.org/) (need download data from [website](https://youtube-vos.org/dataset/download))
```shell
bash test_mask_refine.sh config_davis.json SiamMask_DAVIS.pth ytb_vos 0
```

### Results
These are the reproduction results from this repository. All results can be downloaded from our [project page](http://www.robots.ox.ac.uk/~qwang/SiamMask/).

|                           <sub>Tracker</sub>                           |      <sub>VOT2016</br>EAO /  A / R</sub>     |      <sub>VOT2018</br>EAO / A / R</sub>      |  <sub>DAVIS2016</br>J / F</sub>  |  <sub>DAVIS2017</br>J / F</sub>  |     <sub>Youtube-VOS</br>J_s / J_u / F_s / F_u</sub>     |     <sub>Speed</sub>     |
|:----------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------------------------------:|:------------------------:|
| <sub>[SiamMask-box](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> |       <sub>0.412/0.623/0.233</sub>       |       <sub>0.363/0.584/0.300</sub>       |               - / -              |               - / -              |                      - / - / - / -                       | <sub>**77** FPS</sub> |
| <sub>[SiamMask](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> | <sub>**0.433**/**0.639**/**0.214**</sub> | <sub>**0.380**/**0.609**/**0.276**</sub> | <sub>**0.713**/**0.674**</sub> | <sub>**0.543**/**0.585**</sub> | <sub>**0.602**/**0.451**/**0.582**/**0.477**</sub> |   <sub>56 FPS</sub>   |
| <sub>[SiamMask-LD](http://www.robots.ox.ac.uk/~qwang/SiamMask/)</sub> | <sub>**0.455**/**0.634**/**0.219**</sub> | <sub>**0.423**/**0.615**/**0.248**</sub> | - / - | - / - | - / - / - / - | <sub>56 FPS</sub> |

**Note:** 
- Speed are tested on a NVIDIA RTX 2080. 
- `-box` reports an axis-aligned bounding box from the box branch.
- `-LD` means training with large dataset (ytb-bb+ytb-vos+vid+coco+det).


## Training

### Training Data 
- Download the [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/).
- Preprocess each datasets according the [readme](data/coco/readme.md) files.

### Download the pre-trained model (174 MB)
(This model was trained on the ImageNet-1k Dataset)
```
cd $SiamMask/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training SiamMask base model
- [Setup](#environment-setup) your environment
- From the experiment directory, run
```
cd $SiamMask/experiments/siammask_base/
bash run.sh
```
- Training takes about 10 hours in our 4 Tesla V100 GPUs.
- If you experience out-of-memory errors, you can reduce the batch size in `run.sh`.
- You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)
- After training, you can test checkpoints on VOT dataset.
```shell
bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4  # test all snapshots with 4 GPUs
```
- Select best model for hyperparametric search.
```shell
#bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 8 threads with 4 GPUS
bash test_all.sh -m snapshot/checkpoint_e12.pth -d VOT2018 -n 8 -g 4 # 8 threads with 4 GPUS
```


