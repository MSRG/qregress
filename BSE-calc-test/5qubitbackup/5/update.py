#!/usr/bin/env python
# coding: utf-8



import os, sys, shutil
from glob import glob
import pandas as pd
import json





for i in glob('*/*.json'):
    if 'results' not in i:
        print(i)
        name = os.path.dirname(i)
        logfile = os.path.join(name,'model_log.csv')
        if os.path.exists(logfile):
            df = pd.read_csv(logfile).dropna()
            itermax = df['Iteration'].iloc[-1]
            
            paramfile = os.path.join(name, f'{name}.json')
        
            # Load JSON config
            with open(paramfile, 'r') as f:
                paramdict = json.load(f)
        
            if int(itermax) != 999:
                print(f"Number ran = {int(itermax)}\nCurrent max: {paramdict['MAX_ITER']}")
                paramdict['MAX_ITER'] = 999 - int(itermax)
                print(f"New max: {paramdict['MAX_ITER']}")
        
                # Update JSON file
                with open(paramfile, 'w') as f:
                    json.dump(paramdict, f, indent=4)
        
            # Handle file renaming and enumeration
            existing_files = glob(os.path.join(name, '*_model_log.csv'))
            if existing_files:
                # Get the highest prefix number from existing files
                highest_num = max(
                    int(os.path.basename(f).split('_')[0]) 
                    for f in existing_files 
                    if os.path.basename(f).split('_')[0].isdigit()
                )
                new_filename = os.path.join(name, f"{highest_num + 1}_model_log.csv")
            else:
                # If no other files exist, move to 1_model_log.csv
                new_filename = os.path.join(name, "1_model_log.csv")
        
            shutil.move(logfile, new_filename)
            print(f"Moved {logfile} to {new_filename}")
        else:
            paramfile = os.path.join(name, f'{name}.json')
        
            # Load JSON config
            with open(paramfile, 'r') as f:
                paramdict = json.load(f)      
            
            paramdict['MAX_ITER'] = 1000
            print(f"New max: {paramdict['MAX_ITER']}\n")
            # Update JSON file
            with open(paramfile, 'w') as f:
                json.dump(paramdict, f, indent=4)        
    print()
