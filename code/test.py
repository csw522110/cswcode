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


def read_one_params_data(path):
    with open(path) as f:
        data=np.loadtxt(path, delimiter='\t', dtype=np.double)
    return data

def read_one_nosiy_data(path):
    '''with open(path) as f:
        content=f.read()
        content=content.replace("# star_temp: ","").replace("# star_logg: ","").replace("# star_rad: ","").replace("# star_mass: ","").replace("# star_k_mag: ","").replace("# period: ","")
        content=content.replace("\n", "\t")
        data=np.fromstring(content, sep='\t', dtype=np.double)
    return data'''
    with open(path) as f:
        data=np.loadtxt(path, delimiter='\t', dtype=np.double,comments='#')
        line1=f.readline()
        line1=line1.replace("# star_temp: ","").replace('\n','')
        line2=f.readline()
        line2=line2.replace("# star_logg: ","").replace('\n','')
        line3=f.readline()
        line3=line3.replace("# star_rad: ","").replace('\n','') 
        line4=f.readline()
        line4=line4.replace("# star_mass: ","").replace('\n','') 
        line5=f.readline()
        line5=line5.replace("# star_k_mag: ","").replace('\n','') 
        line6=f.readline()
        line6=line6.replace("# period: ","").replace('\n','')         
    data=data.flatten()
    return np.append(np.array([line1,line2,line3,line4,line5,line6],dtype=np.double),data)

def walkFile(root,mode,file_names):
    data_list=[]
    count=1
    for f_name in file_names:
        #print("%s,times=%d"%(mode,count))
        count=count+1
        path=root+'/'+f_name
        if mode == "noisy":
            data=read_one_nosiy_data(path)
        elif mode == "params":
            data=read_one_params_data(path)
        data_list.append(data)
    return np.array(data_list)
            


train_root_params="D:/ml/ml_data_challenge_database/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
train_root_noisy="D:/ml/ml_data_challenge_database/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"
test_noisy="D:/ml/ml_data_challenge_database/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
path_train="D:/ml/ml_data_challenge_database/name.txt"
path_test="D:/ml/ml_data_challenge_database/test_name.txt"

df_X_file='D:/ml/df_X'
df_y_file='D:/ml/df_y'
df_test_file='D:/ml/test_X'

random_choice=[]
for i in range(1256):
    random_choice.append(random.randint(0,99))
    random_choice.append(random.randint(0,99))
    random_choice.append(random.randint(0,99))
    random_choice.append(random.randint(0,99))

random_X=np.zeros((1256*4,16506))
count=0
for f in os.listdir(df_X_file):
    path=df_X_file+'/'+f
    with open(path, "rb") as f:
        df_X=pickle.load(f)
    for j in range(4):
        random_X[count+j]=df_X.loc[[random_choice[count+j]],:].values
    count=count+4


random_X=pd.DataFrame(random_X)
print(random_X)
with open('D:/ml/random_train_x.pk','wb')as f:
    pickle.dump(random_X, f, protocol = 4)

random_y=np.zeros((1256*4,55))
count=0
for f in os.listdir(df_y_file):
    path=df_y_file+'/'+f
    with open(path, "rb") as f:
        df_y=pickle.load(f)
    for j in range(4):
        random_y[count+j]=df_y.loc[[random_choice[count+j]],:].values
    count=count+4
random_y=pd.DataFrame(random_y)
with open('D:/ml/random_train_y.pk','wb')as f:
    pickle.dump(random_y, f, protocol = 4)






