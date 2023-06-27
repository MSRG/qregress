import pandas
import click
import joblib
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_file(file: str, y_label: str, length: int | float):
    """
    Loads the file given using joblib. Target file must be a joblib output file of type .bin containing a pandas
    dataframe object.

    TODO: Add support for either .csv or .bin joblib files.
    """
    print(f'Loading file {file}... ')
    df = joblib.load(file)

    if length is None:
        df = df.sample(frac=1)

    elif length.is_integer():
        df = df.sample(n=length)

    elif not length.is_integer():
        df = df.sample(frac=length)

    y = df[y_label]
    y = np.array(y)
    x = df.drop([y_label], axis=1)
    x = np.array(x)

    print(f'File successfully loaded and shuffled into {len(y)} samples. ')
    return x, y


def split(x, y, x_dim: int, train_ratio: float):

    if x_dim is not None:
        raise NotImplementedError('PCA / dimension reduction is not yet implemented.')
    if x_dim is None:
        raise ValueError('You must specify a feature dimension. Lower dimension is recommended for training time. ')
    scaler = MinMaxScaler
    x_scaler = scaler((-1, 1))
    y_scaler = scaler((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio)
    sc_X_tr = x_scaler.fit_transform(X_train)
    sc_X_te = x_scaler.transform(X_test)
    sc_y_tr = y_scaler.fit_transform(y_train).reshape(-1)
    sc_y_te = y_scaler.transform(y_test).reshape(-1)
    print(f'Data successfully split into a {train_ratio} train ratio and scaled to {(-1, 1)} ')
    return sc_X_tr, sc_y_tr, sc_X_te, sc_y_te  # returns (X_train y_train X_test y_test)


@click.command()
@click.option('--train_ratio', default='0.8', help='Ratio of dataset to be reserved for training. Must be '
                                                   'interpretable as a float. ')
@click.option('--x_dim', default='16', help='Integer size of feature space to reduce to. ')
@click.option('--length', default=None, help='Number of datapoints to be included in the dataset generation. Can be '
                                             'passed as an int or a ratio. ')
@click.option('--y_label', default='BSE', help='Specify the label of the column to use as target values. ')
@click.option('--file', required=True, help='Source file to use for dataset generation. Must be of .bin format '
                                            'containing pandas dataframe. ')
@click.option('--save_name', default=None, help='Specify the name of the output files. ')
def main(train_ratio, x_dim, length, y_label, file, save_name):
    if save_name is None:
        save_name = y_label
    try:
        train_ratio = float(train_ratio)
    except ValueError:
        print('Could not convert input into float for train ratio. Proceeding with a ratio of 0.8... ')
        train_ratio = 0.8
    try:
        x_dim = int(x_dim)
    except ValueError:
        print('Could not convert input into int for x_dim. Proceeding with size 16... ')
        x_dim = 16
    try:
        if length is not None:
            length = float(length)
    except ValueError:
        print('Could not convert input for length. Proceeding with entire dataset... ')
        length = None

    x, y = load_file(file, y_label, length)

    X_train, y_train, X_test, y_test = split(x, y, x_dim, train_ratio)

    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    y_test = y_test.tolist()

    train = {
        'X': X_train,
        'y': y_train
    }

    test = {
        'X': X_test,
        'y': y_test
    }

    with open(save_name+'_train+.json', 'w') as outfile:
        joblib.dump(train, outfile)
    with open(save_name+'_test+.json', 'w') as outfile:
        joblib.dump(test, outfile)

    print(f'Successfully created outfiles as {save_name}_train.json and {save_name}_test.json')


if __name__ == '__main__':
    main()
