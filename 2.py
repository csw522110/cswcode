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
    for f_name in file_names:
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

f=open(path_train,'r')
name_lst_all=f.readlines()
for i in range(len(name_lst_all)):
    name_lst_all[i]=name_lst_all[i].strip('\n')

f=open("D:/ml/train_year.txt")
train_x_year=f.readlines()
for i in range(len(train_x_year)):
    train_x_year[i]=train_x_year[i].strip('\n')


mean_y=np.zeros((1256,55))
for i in range(1256):
    name_lst=name_lst_all[i*100:i*100+100]
    data_list=walkFile(train_root_params,"params",name_lst)
    df_y=pd.DataFrame(data_list)
    #df_y["name"]=name_lst
    print("%d times,get df_y"%(i+1))
    with open(df_y_file+'/'+train_x_year[i]+'.pk', "wb") as f:
        pickle.dump(df_y, f, protocol = 4) 

    mean_y[i]=df_y.mean(axis=0)

mean_y=pd.DataFrame(mean_y)
with open('D:/ml/mean_train_y.pk','wb')as f:
    pickle.dump(mean_y, f, protocol = 4)


mean_x=np.zeros((1256,16506))
for i in range(1256): 
    name_lst=name_lst_all[i*100:i*100+100]   
    data_list=walkFile(train_root_noisy,"noisy",name_lst)
    df_X=pd.DataFrame(data_list)
    #df_X["name"]=name_lst
    print("%d times,get df_X"%(i+1))
    with open(df_X_file+'/'+train_x_year[i]+'.pk', "wb") as f:
        pickle.dump(df_X, f, protocol = 4)
    
    mean_x[i]=df_X.mean(axis=0)
mean_x=pd.DataFrame(mean_x)
with open('D:/ml/mean_train_x.pk','wb')as f:
    pickle.dump(mean_x, f, protocol = 4)

f=open(path_test,'r')
name_lst_all=f.readlines()
for i in range(len(name_lst_all)):
    name_lst_all[i]=name_lst_all[i].strip('\n')

n=5000
for i in range(11): 
    name_lst=name_lst_all[i*n:min(i*n+n,53900)]   
    data_list=walkFile(test_noisy,"noisy",name_lst)
    df_X=pd.DataFrame(data_list)
    #df_X["name"]=name_lst
    print("%d times,get test_X"%(i+1))
    with open(df_test_file+'/'+str(i)+'.pk', "wb") as f:
        pickle.dump(df_X, f, protocol = 4)


with open('D:/ml/random_train_x.pk','rb')as f:
    x_train=pickle.load(f)
with open('D:/ml/random_train_y.pk','rb')as f:
    y_train=pickle.load(f)
DF21=CascadeForestRegressor(n_jobs=129)
DF21.fit(x_train.values,y_train.values)

for i in range(11): 
    with open(df_test_file+'/'+str(i)+'.pk', "rb") as f:
        df_X=pickle.load(f)
    y_pred=DF21.predict(df_X.values)
    with open("./df21/answer7.txt",'ab') as f:
        np.savetxt(f,X=y_pred,fmt='%.12f',delimiter='\t')
    print("%d times,write y_pred"%(i))
        


