import click
import joblib
import os
import umap
import numpy as np
import pandas as pd
from typing import Union
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


############################################
# Load and Split
############################################


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


def shuffle(df: pd.DataFrame, y_label: str, length: Union[int, float]):
    """
    Shuffles the dataframe and splits it into seperate X and y arrays. Additionally, trims the length of the
    dataset to specified length, either as an int or a ratio of the full set.
    """

    print(f'Shuffling dataframe into x and y sets... ')
    if length is None:
        df = df.sample(frac=1)

    elif length.is_integer():
        length = int(length)
        df = df.sample(n=length)

    elif not length.is_integer():
        df = df.sample(frac=length)

    y = df[y_label]
    y = np.array(y)
    x = df.drop([y_label], axis=1)
    x = np.array(x)

    print(f'Dataframe successfully shuffled into {len(y)} samples. ')
    return x, y


def split(x, y, train_ratio: float, test_ratio: float, validate_ratio: float):
    """
    Splits the dataset into train and test sets then scales each to (-1, 1). Then applies PCA
    to reduce the number of features to x_dim.
    """

    print('Now splitting and scaling data... ')
    # I don't know if we need to create a validate set here or if it should be done somewhere else.

    scaler = MinMaxScaler
    x_scaler = scaler((-1, 1))
    y_scaler = scaler((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio)
    if validate_ratio != 0:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validate_ratio))
    else:
        X_val = np.empty(x.shape)
        y_val = np.empty(y.shape)
    y_train, y_test, y_val = y_train.reshape(-1, 1), y_test.reshape(-1, 1), y_val.reshape(-1, 1)
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    X_val = x_scaler.transform(X_val)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    y_val = y_scaler.transform(y_val)
    print(f'Data successfully split into a {train_ratio} train ratio and scaled to {(-1, 1)} ')

    return X_train, y_train, X_test, y_test, X_val, y_val, y_scaler  # returns (X_train y_train X_test y_test X_val
    # y_val, y_scaler)


############################################
# Dimension Reduction
############################################

dim_methods_list = ['UMAP', 'TSNE', 'PCA']


def dim_reduction(X_train, X_test, X_val, x_dim: int, method: str):
    dim_methods = {
        'UMAP': umap.UMAP(n_components=x_dim),
        'TSNE': TSNE(n_components=x_dim, perplexity=50),
        'PCA': PCA(n_components=x_dim, svd_solver='full')
    }
    print(f'Now applying {method} to reduce to {x_dim} features... ')
    reducer = dim_methods[method]
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.fit_transform(X_test)
    print(f'Successfully reduced train and test. ')
    if not (X_val.base is None):
        print('is inside')
        X_val = reducer.fit_transform(X_val)
    print(f'Successfully reduced the dataset to {x_dim} features. ')

    return X_train, X_test, X_val


def plot_dimension(X_train, X_test, X_val, y_train, y_test, y_val):
    methods = dim_methods_list
    datasets = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test),
        'Validation': (X_val, y_val)
    }

    plt.figure(figsize=(15, 9))
    for i, method in enumerate(methods):
        for j, (dataset, (X_data, y_data)) in enumerate(datasets.items()):
            if dataset == 'Validation' and X_val.base is None:
                continue

            if dataset == 'Train':
                X_train_reduced, _, _ = dim_reduction(X_train, X_test, X_val, x_dim=2, method=method)
                X_data_reduced = X_train_reduced
            elif dataset == 'Test':
                _, X_test_reduced, _ = dim_reduction(X_train, X_test, X_val, x_dim=2, method=method)
                X_data_reduced = X_test_reduced
            else:
                _, _, X_val_reduced = dim_reduction(X_train, X_test, X_val, x_dim=2, method=method)
                X_data_reduced = X_val_reduced

            ax = plt.subplot(len(methods), len(datasets), i * len(datasets) + j + 1)
            ax.scatter(X_data_reduced[:, 0], X_data_reduced[:, 1], c=y_data, cmap='coolwarm', s=1)
            ax.set_title(f'{method} - {dataset} Data')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(ax.scatter(X_data_reduced[:, 0], X_data_reduced[:, 1], c=y_data, cmap='coolwarm', s=1), ax=ax)

    plt.tight_layout()
    plt.show()
    exit()

############################################
# Main
############################################


@click.command()
@click.option('--train_ratio', default=0.8, help='Ratio of dataset to be reserved for training. ')
@click.option('--test_ratio', default=None, type=float, help='Ratio of dataset to reserve for testing. ')
@click.option('--validate_ratio', default=None, type=float, help='Ratio of dataset to reserve for validation. ')
@click.option('--x_dim', default=None, type=int, help='Size of feature space to reduce to. ')
@click.option('--length', default=None, type=float, help='Number of datapoints to be included in the dataset '
                                                         'generation. Can be passed as an int or a ratio. ')
@click.option('--y_label', default='BSE', help='Specify the label of the column to use as target values. ')
@click.option('--dimension_analysis', default='UMAP', type=click.Choice(dim_methods_list),
              help=f'Specify the method to reduce the feature space. ')
@click.option('--visualize', default=False, help='Generates figures of the dimensionality reduction. ')
@click.option('--file', required=True, type=click.Path(exists=True), help='Source file to use for dataset generation. '
                                                                          'Supports either .bin or .csv. ')
@click.option('--save_name', default=None, type=click.Path(), help='Specify the name of the output files. ')
def main(train_ratio, test_ratio, validate_ratio, x_dim, length, y_label, file, save_name, dimension_analysis,
         visualize):
    """
    Performs PCA dimension reduction to specified size and shuffles the data into specified length. Splits the dataset
    into train and test sets and creates seperate files for each.
    """
    if test_ratio is None and validate_ratio is None:
        test_ratio = 1 - train_ratio
        validate_ratio = 0
    elif validate_ratio is None:
        validate_ratio = 1 - train_ratio - test_ratio
    elif test_ratio is None:
        test_ratio = 1 - train_ratio - validate_ratio

    ratio_sum = train_ratio + test_ratio + validate_ratio
    if ratio_sum != 1:
        raise ValueError(f'Train test validate split must sum to 1 but instead got: {ratio_sum}')

    if save_name is None:
        filebase = os.path.basename(file)
        filename, ext = os.path.splitext(filebase)
        save_name = filename

    df = load_file(file)
    x, y = shuffle(df, y_label, length)
    X_train, y_train, X_test, y_test, X_val, y_val, y_scaler = split(x, y, train_ratio, test_ratio,
                                                                     validate_ratio)
    if x_dim is not None:
        if visualize:
            plot_dimension(X_train, X_test, X_val, y_train, y_test, y_val)
        X_train, X_test, X_val = dim_reduction(X_train, X_test, X_val, x_dim, dimension_analysis)

    train = {
        'X': X_train,
        'y': y_train
    }

    test = {
        'X': X_test,
        'y': y_test
    }

    validate = {
        'X': X_val,
        'y': y_val
    }
    train_name = save_name + '_train.bin'
    test_name = save_name + '_test.bin'
    validate_name = save_name + '_validate.bin'
    scaler_name = save_name + '_scaler.bin'

    joblib.dump(train, train_name)
    joblib.dump(test, test_name)
    if validate_ratio != 0:
        joblib.dump(validate, validate_name)
    joblib.dump(y_scaler, scaler_name)

    print(f'Successfully created outfiles as {train_name}, {test_name}'
          f'{f" and {validate_name}. " if validate_ratio != 0 else ". "}')


if __name__ == '__main__':
    main()
