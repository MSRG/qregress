#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import sys
import os
import shutil
import joblib
import json


# In[2]:


for p in glob('*/*.json'):
    print(p)
    with open(p,'r') as f:
        olddt=json.load(f)
    
    
    olddt["BATCH_SIZE"]=25
    olddt["NUM_CORES"]=16
    olddt['TOLERANCE']=1e-6
    olddt['MAX_ITER']=1000
    print(olddt)    

    with open(p,'w') as g:
        json.dump(olddt,g)

