#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json


# In[2]:


with open("scores.txt",'r') as f:
    datadct={}
    for line in f.readlines():
        name, dicts=line.split("Model scores")
        namec=name.split('/')[0]
        dictc=dicts.replace("(","").replace(",)","").replace("}.","}").strip(":").replace("{","").replace("}","").split(",")
        data={i.split(":")[0].replace(" ","").strip("'"):float(i.split(":")[1]) for i in dictc}
        datadct[namec]=data


# In[11]:


with open('scores.json', 'w') as fp:
    json.dump(datadct, fp)


# In[ ]:




