# FitNet: Human Fitness Type and Attribute Prediction Network

## Introduction  
This repo is official **[PyTorch](https://pytorch.org)** implementation of **FitNet: Human Fitness Type and Attribute Prediction Network**. The codes are developed under Ubuntu 18.04 and CUDA 10.1. PyTorch 1.3.1 is used.


## Directory  
### Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output 
|-- tool
```  
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for FitNet.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result. 
* `tool` contains data pre-processing code (`get_exercise_dict.py`), which makes `exercise_dict.json`.

  
### Data  
You need to follow directory structure of the `data` as below.  
```  
${ROOT}  
|-- data  
|   |-- Sleek 
|   |-- |-- data
|   |   |   |-- exercise_dict.json
|   |   |   |-- Day01_200921_F
|   |   |   |-- Day02_200922_F
...
|   |   |   |-- Day34_201106_F
```  

### Output  
You need to follow the directory structure of the `output` folder as below.  
```  
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  


## Running FitNet 
### Start  
* Install **[PyTorch](https://pytorch.org)** and Python >= 3.7.3 
* In the `main/config.py`, you can change settings of the model including network backbone and input size and so on.  
* There are two stages. 1) `exer` and 2) `attr`. In the `exer` stage, FitNet is trained to predict exercise type (e.g., push up and benchpress). In the `attr` stage, FitNet is trained to predict exercise attribute (e.g., too fast).
  
### Train  
#### 1. exer stage
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --stage exer --exer_idx -1
```  
to train FitNet in the `exer` stage on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. 

#### 2. attr stage
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --stage attr --exer_idx $EXER_IDX
```  
to train FitNet in the `attr` stage on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. 
`$EXER_IDX` is exercise index, defined in `data/Sleek/data/exercise_dict.json`.

#### Batched training
As there are many number of exercises, I provide batched training script, which trains FitNet on all exercises in `exer` and `attr` stages.
In the `main` folder, run
```bash
python run_train.py`

  
### Test  
Place trained model at the `output/model_dump/`.  Choose the stage you want to test among `exer` and `attr`.
  
In the `main` folder, run  
```bash  
python test.py --gpu 0-3 --stage $STAGE --test_epoch 20 --exer_idx $EXER_IDX
```  
to test FitNet in `$STAGE` stage (should be one of `exer` and `attr`) on the GPU 0,1,2,3 with 20th epoch trained model. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.  
`$EXER_IDX` is -1 when the `$STAGE` is `exer`. Otherwise, choose one from `exercise_dict.json`.

For the batched test, in the `main` folder, run
```bash
python run_test.py
```
  
## Results  
Here I report the performance of the FitNet.

<p align="center">
<img src="assets/capture1.png">
</p>

<p align="center">
<img src="assets/capture2.png">
</p>

<p align="center">
<img src="assets/capture3.png">
</p>

<p align="center">

<img src="assets/capture4.png">
</p>

