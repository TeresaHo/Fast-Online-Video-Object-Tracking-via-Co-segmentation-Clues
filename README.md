### Project: Fast-Online-Video-Object-Tracking-via-Co-segmentation-Clues

### Project description
S2iamMask (Spatial-aware SiamMask) integrates a semantic segmentation branch which can further improve the robustness of the object tracker. Previous methods use mask branch in a way which sacrifices spatial information. However, spatial information is crucial for mask generation. To better address this problem, we proposal a two-stage training strategy and directly use the proposals obtained in first stage as the input for second stage. In addition, fusion layers for FCN  were used to combine semantic  informationfrom a deeper, coarse layer with appearance information from a shallow, fine layer to produce detailed segmentations. Finally, during inference, we design a voting mechanism to prevent the mask from being heavily dependent on the output score.


<div align="center">
  <img src="https://github.com/TeresaHo/VOTCoSeg/blob/master/img/framework.png" width="600px" />
</div>


## Contents
1. [Environment Setup](#environment-setup)
2. [Testing Models](#testing-models)
3. [Training Models](#training-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, RTX 2080 GPUs

- Clone the repository 
```
git clone https://github.com/TeresaHo/VOTCoSeg.git && cd VOTCoSeg/S2iamMask_new
export S2iamMask_new=$PWD
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
cd $S2iamMask_new/data
sudo apt-get install jq
bash get_test_data.sh
```

- Evaluate performance on [VOT](http://www.votchallenge.net/)
```shell
bash test_mask_refine.sh config_vot.json <path to your model weights> VOT2016 0
bash test_mask_refine.sh config_vot.json <path to your model weights> VOT2018 0
python ../../tools/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016
python ../../tools/eval.py --dataset VOT2018 --tracker_prefix C --result_dir ./test/VOT2018
```

### Results (Tested on VOT 2018 dataset)
| Method                   | Accuracy      | Robustness | EAO   | Loss frames |
| ----------------------   | ----------    | --------   | ----- | ----------- |
| S2iamMask                | **0.641**         | **0.195**  | 0.432 | **44**          | 
| S2iamMask                | 0.639         | 0.214      | **0.433** | 52          |



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
cd $S2iamMask_new/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training S2iamMask base model
- [Setup](#environment-setup) your environment
- From the experiment directory, run
```
cd $SsiamMask_new/experiments/siammask_base/
bash run.sh
- After training, you can test checkpoints on VOT dataset.
```shell
bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4  # test all snapshots with 4 GPUs
```
- Select best model for hyperparametric search.
```shell
#bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 8 threads with 4 GPUS
bash test_all.sh -m snapshot/checkpoint_e12.pth -d VOT2018 -n 8 -g 4 # 8 threads with 4 GPUS
```


