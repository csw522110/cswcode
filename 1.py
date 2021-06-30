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
        print("%s,times=%d"%(mode,count))
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

f=open(path_train,'r')
name_lst_all=f.readlines()
for i in range(len(name_lst_all)):
    name_lst_all[i]=name_lst_all[i].strip('\n')


name_lst=random.sample(name_lst_all,2000)

data_list=walkFile(train_root_params,"params",name_lst)
df_y=pd.DataFrame(data_list)
df_y["name"]=name_lst
print("get df_y")
data_list=walkFile(train_root_noisy,"noisy",name_lst)
df_X=pd.DataFrame(data_list)
#df_X["name"]=name_lst
print("get df_X")


DF21=CascadeForestRegressor(n_jobs=129)
DF21.fit(df_X.values, df_y.drop(columns=['name']).values)


f=open(path_test,'r')
name_lst_all=f.readlines()
for i in range(len(name_lst_all)):
    name_lst_all[i]=name_lst_all[i].strip('\n')

all_len=len(name_lst_all)
lst=list(range(0,all_len,1000))
lst.append(all_len)

for i in range(len(lst)-1):
    name_lst=name_lst_all[lst[i]:lst[i+1]]
    data_list=walkFile(test_noisy,"noisy",name_lst)
    df_X=pd.DataFrame(data_list)
    df_X["name"]=name_lst
    print((lst[i],lst[i+1]))
    print("get df_X")
    y_pred=DF21.predict(df_X.drop(columns=['name']).values)
    with open("./df21/answer3.txt",'ab') as f:
        np.savetxt(f,X=y_pred,fmt='%.12f',delimiter='\t')
    print("write y_pred")
