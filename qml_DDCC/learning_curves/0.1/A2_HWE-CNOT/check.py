import logging
from time import time, sleep
logging.basicConfig(level=logging.DEBUG)

from joblib import Parallel, delayed
import os

def work(i):
    sleep(2)
    return f"Process {i} on PID {os.getpid()}"
ran = range(16*4)
t0 = time()
results = Parallel(n_jobs=-1,verbose=-1)(delayed(work)(i) for i in ran)
t1 = time()
print(f"{t1-t0:.4f} s")

t0 = time()
results = [work(i) for i in ran]
t1 = time()
print(f"{t1-t0:.4f} s")

