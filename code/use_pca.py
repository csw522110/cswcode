import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import random
import copy
from sklearn.decomposition import PCA


            
train_root_params="D:/ml/ml_data_challenge_database/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
train_root_noisy="D:/ml/ml_data_challenge_database/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"
test_noisy="D:/ml/ml_data_challenge_database/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
path_train="D:/ml/ml_data_challenge_database/name.txt"
path_test="D:/ml/ml_data_challenge_database/test_name.txt"

df_X_file='D:/ml/df_X'
df_y_file='D:/ml/df_y'
df_test_file='D:/ml/test_X'



with open('D:/ml/random_train_x.pk','rb')as f:
    x_train=pickle.load(f).values
with open('D:/ml/random_train_y.pk','rb')as f:
    y_train=pickle.load(f)
pca=PCA(n_components=5000)
pca=pca.fit(x_train)
x_train=pca.transform(x_train)
DF21=CascadeForestRegressor(n_jobs=129)
DF21.fit(x_train,y_train.values)

for i in range(10): 
    with open(df_test_file+'/'+str(i)+'.pk', "rb") as f:
        df_X=pickle.load(f).values
    pca=pca.fit(df_X)
    df_X=pca.transform(df_X)
    y_pred=DF21.predict(df_X)
    with open("./df21/answer7.txt",'ab') as f:
        np.savetxt(f,X=y_pred,fmt='%.12f',delimiter='\t')
    print("%d times,write y_pred"%(i))
        