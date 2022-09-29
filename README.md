

# HydraGCN
 Multi-Modal data analysis framework for medical applications 
 
 `GCN` `Multi-Modality` `Alzeihmer_Prediction` `COVID-19_Classification` `Skin_Lesion_Detection`
 
## About 
### (Applied Praktikum - Machine Learning in Medical Imaging - TU Munich)

In  medical  research,  multi-modal  datasets  from  large-scale  population-based studies  are  an  essen:al  tool  towards  better  diagnosis  and  treatment  of  disease. Multimodal data comprises imaging and non-imaging data and is available in many domains.  In technical research, such datasets serve as enablers of Computer-Aided Diagnosis (CADx) with machine learning (ML). In this project, we will be focusing on four such multi-modal dataset HCP dataset and UK Biobank for the application of age and gender predictoon. In  this  project,  we  analyze  multi-modal  datasets  with  all  its  challenges  such  as imbalance,  small  size,  missing  values  etc  with  the  perspec:ve  of  graph convolu:onal networks. We will work on a framework developed at CAMP called *‘HydraGCN’*  which  is designed  with  a  multi-head  input  and  output  architecture, which  allows  for  flexible  modeling  of  multimodal  input  data  and  which  enables multi-target learning through separate output heads and losses.



| Task        | Classes  |  Dataset  | Modality  |
| ------------- |:-------------:| :-------------:| :-------------:| 
| Alzeihmer Disease Prediction     | CN - Cognitively normaL,MCI - mild cognitive impairment or AD - probable Alzheimer's Disease | [TADPOLE](https://tadpole.grand-challenge.org/Details/) |  Clinical Features 
| Covid-19 Disease Classification      |  Covid Positive or Negative  |   [Covid-19 iCTCF](https://ngdc.cncb.ac.cn/ictcf/)  |  Clinical Features  + Images
| Skin Lesion Detection      |   Melanocytic nevi, Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, Dermatofibroma  |   [HAM10K](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  |  Clinical Features  + Images




## Graph Convolutional Networks

We have used an ensemble of 4 GCN's
- [x] GCN - Standard GCN
- [x] MGMC - Multi-Graph Matrix Completion
- [x] GAT - Graph Attention Networks
- [x] DGM - Differentiable Graph Module

## Setup

- Create a conda env from the requirements file 

 ``` terminal 
conda create -y --name hydragcn python=3.7
conda install -c conda-forge --file requirements.txt
conda activate hydragcn
  ```
- Create Dataloader 

Update for each dataset the classes in
 `/base/setup/dataset_setup.py` 

- Update the Configurations

Go to `/application/loop_experiments.py` 

 ``` terminal 
# Choose one dataset to work on

dataset_list = ['TADPOLE']
dataset_list = ['COVIDiCTCF']
dataset_list = ['HAM10K']
```

``` terminal 
# Choose the model ensemble to work on

model_list = ['GCN']
model_list = ['FullGAT']
model_list = ['MGMC']
model_list = ['CDGM']
model_list = ['Hydra']
``` 

``` terminal 
# Choose the corresponding yaml configuration OR make your own @'../base/configs/<dataset>/<model_you_like>.yaml'

model_yaml_dir = '../base/configs/TADPOLE/MGMC.yaml'
model_yaml_dir = '../base/configs/COVID/MGMC.yaml'
model_yaml_dir = '../base/configs/COVID/Hydra.yaml'
model_yaml_dir = '../base/configs/HAM10K/Hydra_HAM10K.yaml'
```

## Run the Experiment

``` terminal 
# Starting point of our code 
python /applications/loop_experiments.py
``` 

Problems ? Happy Debugging !
  
