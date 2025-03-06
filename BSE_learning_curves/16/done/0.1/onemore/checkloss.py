import numpy as np
import joblib, json, os, sys
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for i in glob('*/*model_log.csv'):
    df = pd.read_csv(i)
    print(os.path.dirname(i),df['Cost'].iloc[-10:].std())
