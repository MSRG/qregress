import click
import joblib
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_file(file: str):
    """
    Loads the file given using joblib. Target file must be a joblib output file of type .bin containing a pandas
    dataframe object.

    """
    print(f'Loading file {file}... ')

    file_base = os.path.basename(file)
    name, extension = os.path.splitext(file_base)
    if extension == '.bin':
        df = joblib.load(file)
        print(f'File loaded successfully! ')
        return df
    elif extension == '.csv':
        df = pd.read_csv(file)
        print(f'File loaded successfully! ')
        return df


def shuffle(df: pd.DataFrame, y_label: str, length: int | float):
    """
    Shuffles the dataframe and splits it into seperate X and y arrays. Additionally, trims the length of the
    dataset to specified length, either as an int or a ratio of the full set.
    """

    print(f'Shuffling dataframe into x and y sets... ')
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

    print(f'Dataframe successfully shuffled into {len(y)} samples. ')
    return x, y


def split(x, y, x_dim: int, train_ratio: float):
    """
    Splits the dataset into train and test sets then scales each to (-1, 1). Then applies PCA
    to reduce the number of features to x_dim.
    """

    print('Now splitting and scaling data... ')
    if x_dim is None:
        raise ValueError('You must specify a feature dimension. ')
    scaler = MinMaxScaler
    x_scaler = scaler((-1, 1))
    y_scaler = scaler((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    X_tr = x_scaler.fit_transform(X_train)
    X_te = x_scaler.transform(X_test)
    y_tr = y_scaler.fit_transform(y_train)
    y_te = y_scaler.transform(y_test)
    print(f'Data successfully split into a {train_ratio} train ratio and scaled to {(-1, 1)} ')

    print(f'Now applying PCA to reduce to {x_dim} features... ')
    pca = PCA(n_components=x_dim, svd_solver='full')
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    print(f'Successfully reduced the dataset to {x_dim} features. ')
    return X_tr, y_tr, X_te, y_te  # returns (X_train y_train X_test y_test)


@click.command()
@click.option('--train_ratio', default=0.8, help='Ratio of dataset to be reserved for training. Must be '
                                                 'interpretable as a float. ')
@click.option('--x_dim', default=16, help='Size of feature space to reduce to. ')
@click.option('--length', default=None, type=float, help='Number of datapoints to be included in the dataset '
                                                         'generation. Can be passed as an int or a ratio. ')
@click.option('--y_label', default='BSE', help='Specify the label of the column to use as target values. ')
@click.option('--file', required=True, type=click.Path(exists=True), help='Source file to use for dataset generation. '
                                                                          'Supports either .bin or .csv. ')
@click.option('--save_name', default=None, type=click.Path(), help='Specify the name of the output files. ')
def main(train_ratio, x_dim, length, y_label, file, save_name):
    if save_name is None:
        filebase = os.path.basename(file)
        filename, ext = os.path.splitext(filebase)
        save_name = filename

    df = load_file(file)
    x, y = shuffle(df, y_label, length)
    X_train, y_train, X_test, y_test = split(x, y, x_dim, train_ratio)

    train = {
        'X': X_train,
        'y': y_train
    }

    test = {
        'X': X_test,
        'y': y_test
    }
    train_name = save_name + '_train.bin'
    test_name = save_name + '_test.bin'

    joblib.dump(train, train_name)
    joblib.dump(test, test_name)

    print(f'Successfully created outfiles as {train_name} and {test_name} ')


if __name__ == '__main__':
    main()
