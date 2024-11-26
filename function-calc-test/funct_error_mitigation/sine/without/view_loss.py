#!/bin/python3
import click
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option('--path', required=True, type=click.Path(exists=True), help='Path to model_log.csv')

def main(path):

    with open(path,'r') as f:
        filename=f.readlines()
    
    save=[]
    for i in filename:
        if ':' in i:
            splitf=i.split(',')
            save.append((splitf[1],splitf[2]))
    
    save=np.array(save).astype(float)
    
    plt.plot(save[:,0],save[:,1])
    plt.show()


if __name__ == '__main__':
    main()
