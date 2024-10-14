import os,sys,joblib
from glob import glob

for i in sorted(glob("*_DDCC_train.bin")):
    with open(i,'rb') as g:
        data = joblib.load(g)
    X=data['X']
    y=data['y']
    print(i,X.shape,y.shape)
    print(X.min(),X.max())
    print(y.min(),y.max())
    print()
