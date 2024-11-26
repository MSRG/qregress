#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# !{sys.executable} -m pip install shap
import joblib
import time
import psi4
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
from helper_CC_ML_spacial import *
# SHAP
import shap

# In[ ]:


# mol = psi4.geometry("""
# 0 1
# O 0.769120 1.020656 0.000000
# O 0.000000 0.000000 0.000000
# O 0.769120 -1.020656 0.000000
# symmetry c1
#         """)
        



# psi4.core.clean()
# psi4.core.be_quiet()
# psi4.set_options({'basis':'3-21G',
# # psi4.set_options({'basis':'STO-3G',
#                   'scf_type':     'pk',
#                   'reference':    'rhf',
#                   'mp2_type':     'conv',
#                   'e_convergence': 1e-8,
#                   'd_convergence': 1e-8})
# rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
# scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
# A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=True)

# A.compute_energy()


# In[ ]:


properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']


# In[ ]:


data_dict={}
for struct in glob('./ddcc-voglab2019/water/*xyz'):


    print(struct)
    with open(struct,'r') as f:
        text=f.read()
    
    xyz=False
    if xyz==True: 
        qmol = psi4.qcdb.Molecule.from_string(text, dtype='xyz')
        mol = psi4.geometry(qmol.create_psi4_string_from_molecule()+ 'symmetry c1')                
    else:                                
        mol = psi4.geometry(text)  

    psi4.core.clean()
    psi4.core.be_quiet()
    # psi4.set_options({'basis':'3-21G',
    psi4.set_options({'basis':'STO-3G',
                      'scf_type':     'pk',
                      'reference':    'rhf',
                      'mp2_type':     'conv',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    try:
        rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
        scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
        A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=True)
        
        A.compute_energy()
        
        data=pd.DataFrame(np.array([getattr(A,attr).flatten() for attr in properties]).T,columns=properties)
        data_dict[struct.split('_')[0]]=data
    except:
        pass

for k,v in sorted(data_dict.items()):
    print(k)
    v.to_pickle(f"./data/{os.path.basename(k.replace('.xyz',''))}.pkl.gz", compression='gzip')


# In[ ]:


data_dict={int(''.join(filter(str.isdigit, v.split('/')[-1].split('.')[0]))):pd.read_pickle(v,compression='gzip') for v in glob('./data/*.pkl.gz')}


# In[ ]:


train,test=train_test_split(list(data_dict.keys()),train_size=0.8,test_size=0.2)


# In[ ]:


X_train=[]
y_train=[]
X_test=[]
y_test=[]

X_train=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in train])
y_train=np.hstack([data_dict[i]['t2'].to_numpy() for i in train])

X_test=np.vstack([data_dict[i].drop(columns=['t2']).to_numpy() for i in test])
y_test=np.hstack([data_dict[i]['t2'].to_numpy() for i in test])


scaler = MinMaxScaler
x_scaler = scaler((-1, 1))
y_scaler = scaler((-1, 1))

X_train=x_scaler.fit_transform(X_train)
y_train=y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

X_test=x_scaler.transform(X_test)
y_test=y_scaler.transform(y_test.reshape(-1,1)).flatten()


# In[ ]:


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[ ]:


rfr=RandomForestRegressor()
params={
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
grid_search = GridSearchCV(rfr, params, cv=5,n_jobs=-1,verbose=1000)


print(f'Now fitting model... ')
st = time.time()
grid_search.fit(X_train,y_train)
print(f'Completed fitting model in {time.time() - st:.4f} seconds. ')

model = grid_search.best_estimator_


# In[ ]:


models={}
# model = KNeighborsRegressor(n_neighbors=1,n_jobs=-1)
model.fit(X_train, y_train)
print(model.score(X_train,y_train),model.score(X_test,y_test))
models['original']={'Train':model.score(X_train,y_train),'Test':model.score(X_test,y_test)}


# In[ ]:





# In[ ]:


explainer = shap.Explainer(model.predict, np.vstack([X_train,X_test]),n_jobs=-1,feature_names=properties[:-1])
shap_values = explainer(X_test)
shap.plots.bar(shap_values,max_display=16)


# In[ ]:





# In[ ]:





# In[ ]:


top5=np.argsort(shap_values.abs.values.mean(axis=0))[-5:]
top16=np.argsort(shap_values.abs.values.mean(axis=0))[-16:]
top5_names=np.array(properties)[top5]
top16_names=np.array(properties)[top16]


# In[ ]:


with open(f'5_DDCC_train.bin','wb') as f:
    joblib.dump({'X':X_train[:,top5],'y':y_train},f)
with open(f'5_DDCC_test.bin','wb') as f:
    joblib.dump({'X':X_test[:,top5],'y':y_test},f)
with open(f'5_DDCC_scaler.bin','wb') as f:
    joblib.dump(y_scaler,f)








with open(f'16_DDCC_train.bin','wb') as f:
    joblib.dump({'X':X_train[:,top16],'y':y_train},f)
with open(f'16_DDCC_test.bin','wb') as f:
    joblib.dump({'X':X_test[:,top16],'y':y_test},f)
with open(f'16_DDCC_scaler.bin','wb') as f:
    joblib.dump(y_scaler,f)


# In[ ]:


model.fit(X_train[:,top5],y_train)
# model.score(X_train[:,top5],y_train),model.score(X_test[:,top5],y_test)
models['5']={'Train':model.score(X_train[:,top5],y_train),'Test':model.score(X_test[:,top5],y_test)}


model.fit(X_train[:,top16],y_train)
# model.score(X_train[:,top5],y_train),model.score(X_test[:,top5],y_test)
models['16']={'Train':model.score(X_train[:,top16],y_train),'Test':model.score(X_test[:,top16],y_test)}


# In[ ]:


SMALL_SIZE = 12
MEDIUM_SIZE = SMALL_SIZE
BIGGER_SIZE = SMALL_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

g=sns.barplot(data=pd.DataFrame.from_dict(models).reset_index().melt(id_vars='index'),x='variable',y='value',hue='index',palette=sns.color_palette('Paired',2))
for container in g.containers:
    g.bar_label(container, fmt='%0.4f')
    
plt.ylim(0,1.1)
plt.legend(loc=4,framealpha=1)
plt.ylabel('R$^{2}$')
plt.xlabel('Feature Set Dimensions')
# plt.title('DDCC Feature Set')
plt.tight_layout()
plt.savefig('DDCC_feature_set.png',dpi=300,bbox_inches='tight')

