# SCRIPT TO REMOVE NA VALUES IN AGE AND ADDING AN IMAGE PATH COLUMNS TO CSV FILE

import pandas as pd
from glob import glob
import os 
import numpy as np
from PIL import Image


meta_path = "C:/Users/kamranisg/Desktop/MLMI/supergcn_2.0-feature-mlmi_ss21_alex/data/ham10k/HAM10000_metadata" # add your ham10k metadata local path

meta_data = pd.read_csv(meta_path)

meta_data['age'].fillna(int(meta_data['age'].mean()),inplace=True)
base_skin_dir="C:\\Users\\kamranisg\\Desktop\\MLMI\\supergcn_2.0-feature-mlmi_ss21_alex\\data\\ham10k\\ham10k_download" # change to your dir
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
m = pd.read_csv(os.path.join("C:\\Users\\kamranisg\\Desktop\\MLMI\\supergcn_2.0-feature-mlmi_ss21_alex\\data\\ham10k",'HAM10000_metadata'))
meta_data['path']=meta_data['image_id'].map(imageid_path_dict.get)	        
meta_data['image']=meta_data['path'].map(lambda x: np.asarray(Image.open(x).resize((125,100))))

path = "C:\\Users\\kamranisg\\Desktop\\MLMI\\supergcn_2.0-feature-mlmi_ss21_alex\\data\\ham10k\\HAM10K_FINAL" # NEW HAM10K  meta file path
with open(path,"w") as f:
	meta_data.to_csv("C:\\Users\\kamranisg\\Desktop\\MLMI\\supergcn_2.0-feature-mlmi_ss21_alex\\data\\ham10k\\HAM10K_FINAL") #NEW HAM10K meta file path
