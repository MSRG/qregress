import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_log.csv",header=None)


df['Time'] = pd.to_datetime(df[0], format='%a %b %d %H:%M:%S %Y')
df['total time']=[0]*len(df)
for idx,t in enumerate(df['Time'].to_numpy()):
    print((t-df['Time'].iloc[0]).total_seconds())
    df['total time'].iloc[idx]=(t-df['Time'].iloc[0]).total_seconds()

print(df)

plt.plot(df['total time'],df[1])
plt.show()
