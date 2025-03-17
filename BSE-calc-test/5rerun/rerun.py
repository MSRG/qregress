#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob
import pandas as pd
import numpy as np
import subprocess
import json


# In[2]:


def update_json(filename,iters):
    with open(filename,'r') as f:
        olddt=json.load(f)
    
    olddt['MAX_ITER']=iters
    print(olddt)    

    with open(filename,'w') as g:
        json.dump(olddt,g)


# In[3]:


topdir = os.getcwd()


# In[4]:


def clean(path,name):
    files = f'{name}_encoder.svg', f'{name}_plot.svg', f'{name}.out', f'1_model_log.csv', f'model_log.csv', f'final_state_model.bin', f'{name}_ansatz.svg', f'{name}_predicted_values.csv', f'{name}_results.json'
    for i in files:
        if os.path.exists(os.path.join(path,i)):
            os.remove(i)


# In[12]:


check = {}
for i in glob('*/*.json'):
    if 'results' not in i:
        dirname = os.path.dirname(i)
        dirpath = os.path.join(topdir,dirname)
        # print(dirname)
        
        check[dirname] = np.nansum([pd.read_csv(m).dropna()['Iteration'].max() for m in glob(dirpath+"/*model_log.csv")])
            


# In[13]:


def run(topdir,path,name,continuefile=None):
    if continuefile==None:
        return f"""python {topdir}/main.py --save_path {path} --settings {path}/{name}.json \
--train_set {topdir}/PCA5_0.8_Morgan_train.bin --test_set {topdir}/PCA5_0.8_Morgan_test.bin \
--scaler {topdir}/PCA5_0.8_Morgan_scaler.bin --save_circuits True  > {path}/{name}.out 2>&1"""
    else:
        return f"""python {topdir}/main.py --save_path {path} --settings {path}/{name}.json \
--train_set {topdir}/PCA5_0.8_Morgan_train.bin --test_set {topdir}/PCA5_0.8_Morgan_test.bin \
--scaler {topdir}/PCA5_0.8_Morgan_scaler.bin --save_circuits True \
--resume_file {continuefile} > {path}/{name}.out 2>&1"""


# In[17]:


for k,v in sorted(check.items()):

    path = os.path.join(topdir,k)
    print(k,v)
    update_json(os.path.join(path,k+".json"),1000)
    # Run the command
    subprocess.run(run(topdir,path,k), shell=True, check=True)
