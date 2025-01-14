#!/usr/bin/env python3
import click
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import matplotlib

@click.command()
@click.option('--path', required=True, type=click.Path(exists=True), help='Path to model_log.csv; e.g. A2_HWE-CNOT/')
def main(path):
    splitsort = []
    smaller = []
    for i in glob(os.path.join(path, '*log.csv')):
        splitname = os.path.basename(i).split('_')
        if len(splitname)==3:
            splitsort.append((int(splitname[0]),i))
        else:
            smaller.append(i)

    loss = []
    for i in list(np.array(sorted(splitsort))[:,1])+smaller:
        print(f"Processing file: {i}")
        loss.append(split(i)[:, 1])  # Get only the loss values (assumed to be in the second column)
    plt.plot(np.hstack(loss))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

def split(path):
    """Helper function to read CSV file and extract required columns."""
    save = []
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                split_line = line.strip().split(',')
                # Assuming the structure of each line is: [timestamp, loss_value, ...]
                save.append((float(split_line[1]), float(split_line[2])))
    return np.array(save)

if __name__ == '__main__':
    main()

