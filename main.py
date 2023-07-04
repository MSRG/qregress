import joblib
import click
import json
import time
import os
import itertools
import warnings
import collections.abc

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from sklearn.metrics import mean_squared_error, r2_score
from qiskit_ibm_provider import IBMProvider

from quantum.Quantum import QuantumRegressor
from settings import ANSATZ_LIST, ENCODER_LIST


# Global variables
OPTIMIZER = None
SHOTS = None
X_DIM = None
BACKEND = None
DEVICE = None
SCALE_FACTORS = None
ANSATZ = None
ENCODER = None
POSTPROCESS = None
ERROR_MITIGATION = None
LAYERS = None
PROVIDER = None
TOKEN = None
HYPERPARAMETERS = None


############################################
#  Lists of acceptable values
############################################


ERROR_MITIGATION_LIST = [
    None,
    'MITIQ_LINEAR',
    'MITIQ_Richardson',
    'M3',
    'TREX'
]

POSTPROCESS_LIST = {
    'None': None,
    'simple': 'simple',
    'ridge': 'ridge',
    'lasso': 'lasso'
}


############################################
# Utility functions
############################################


def parse_settings(settings_file):
    with open(settings_file, 'r') as fp:
        settings = json.load(fp)

    global OPTIMIZER
    OPTIMIZER = settings['OPTIMIZER']

    global SHOTS
    SHOTS = settings['SHOTS']

    global BACKEND
    BACKEND = settings['BACKEND']

    global DEVICE
    DEVICE = settings['DEVICE']

    global SCALE_FACTORS
    SCALE_FACTORS = settings['SCALE_FACTORS']

    global POSTPROCESS
    POSTPROCESS = settings['POSTPROCESS']

    global ERROR_MITIGATION
    ERROR_MITIGATION = settings['ERROR_MITIGATION']

    global LAYERS
    LAYERS = settings['LAYERS']

    global HYPERPARAMETERS
    HYPERPARAMETERS = settings['HYPERPARAMETERS']

    # classes aren't JSON serializable, so we store the key in the settings file and access it here.
    global ANSATZ
    ANSATZ = ANSATZ_LIST[settings['ANSATZ']]

    global ENCODER
    ENCODER = ENCODER_LIST[settings['ENCODER']]


def load_dataset(file):
    print(f'Loading dataset from {file}... ')
    data = joblib.load(file)
    X = data['X']
    y = data['y']

    global X_DIM
    _, X_DIM = X.shape
    print(f'Successfully loaded {file} into X and y data. ')
    return X, y


def save_token(instance, token):
    global PROVIDER
    PROVIDER = IBMProvider(instance=instance)
    global TOKEN
    TOKEN = token
#    QiskitRuntimeService.save_account(channel='ibm_quantum', token=token, overwrite=True)


############################################
# Main
############################################

@click.command()
@click.option('--settings', required=True, type=click.Path, help='Settings file for running ML. ')
@click.option('--train_set', required=True, type=click.Path, help='Datafile for training the ML model. ')
@click.option('--test_set', default=None, type=click.Path, help='Optional datafile to use for testing and scoring the '
                                                                'model. ')
@click.option('--instance', default=None, help='Instance for running on IBMQ devices. ')
@click.option('--token', default=None, help='IBMQ token for running on hardware. ')
@click.option('--save_model', default=False, help='Whether to save the trained model to file. ')
@click.option('--save_circuits', default=False, help='Whether to save a figure of encoder and ansatz circuits. ')
@click.option('--title', default=None, help='Title to use for save files. If none, infers it from settings file. ')
def main(settings, train_set, test_set, instance, token, save_model, save_circuits, title):
    X_train, y_train = load_dataset(train_set)
    parse_settings(settings)
    if DEVICE == 'qiskit.ibmq':
        save_token(instance, token)
    kwargs = create_kwargs()

    if title is None:
        title = os.path.basename(settings)
        title, _ = os.path.splitext(title)
        title = title + '_model.bin'
    else:
        title = title + '_model.bin'

    if save_circuits:
        plot_circuits(title)

    if test_set is not None:
        X_test, y_test = load_dataset(test_set)
    else:
        X_test, y_test = None, None

    print(f'Training model with dataset {train_set} \n at time {time.asctime()}... ')
    st = time.time()
    model, hyperparams, score, results = grid_search(QuantumRegressor, HYPERPARAMETERS, X_train, y_train,
                                                     X_test, y_test, **kwargs)

    et = time.time()
    print(f'Training complete taking {st-et} total seconds. Best hyperparameters found to be {hyperparams}. ')

    if save_model:
        joblib.dump(model, title)

    if test_set is not None:
        evaluate(model, X_train, X_test, y_train, y_test, plot=True, title=title)


def plot_circuits(title):
    draw_ansatz = qml.draw_mpl(ANSATZ)
    draw_ansatz(np.random.rand(ANSATZ.num_params))
    plt.savefig(title+'_ansatz.svg')

    draw_encoder = qml.draw_mpl(ENCODER)
    draw_encoder(np.random.rand(X_DIM))
    plt.savefig(title+'_encoder.svg')


def create_kwargs():
    #  First have to apply specific ansatz settings: setting number of layers and the number of wires based on features
    ANSATZ.layers = LAYERS
    ANSATZ.set_wires(range(X_DIM))

    kwargs = {
        'encoder': ENCODER,
        'variational': ANSATZ,
        'num_qubits': X_DIM,
        'optimizer': OPTIMIZER,
        'device': DEVICE,
        'backend': BACKEND,
        'postprocess': POSTPROCESS,
        'error_mitigation': ERROR_MITIGATION,
        'provider': PROVIDER,
        'token': TOKEN
    }
    return kwargs


def grid_search(model, hyperparameters: dict, x_train, y_train, x_test=None, y_test=None, **kwargs):
    """
    Performs a grid search on the given model. Trains the model for each combination of hyperparameters, and then
    trains it using x_train, y_train. Scores each model using r2_score on the test dataset and returns the best
    performing model with its score and hyperparameters. Any additional parameters to be passed to the model are
    handled with kwargs.

    :return: trained_model, dict: best_hyperparameters, flaot: best_score, list: results
    """
    for x in hyperparameters.values():
        if not isinstance(x, collections.abc.Sequence):
            raise ValueError('Dictionary must contain list-like objects of values to try! ')

    results = []
    best_score = float('-inf')
    best_model = None
    best_hyperparameters = {}

    param_combinations = list(itertools.product(*hyperparameters.values()))
    for combination in param_combinations:
        update = dict(zip(hyperparameters.keys(), combination))
        kwargs.update(update)
        print(f'Beginning training with hyperparameters f={hyperparameters["f"]}, alpha={hyperparameters["alpha"]}, '
              f'beta={hyperparameters["beta"]}... ')
        st = time.time()
        built_model = model(**kwargs)
        built_model.fit(x_train, y_train, callback_interval=1)
        if x_test is not None and y_test is not None:
            y_pred = built_model.predict(x_test)
            score = r2_score(y_test, y_pred)
            results.append(score)
        else:
            warnings.warn('Using train set for hyperparameter search may lead to overfitting. ')
            y_pred = built_model.predict(x_train)
            score = r2_score(y_train, y_pred)
            results.append(score)

        if score > best_score:
            print(f'Training complete taking {st-time.time()} seconds. Saving model as new best. ')
            best_score = score
            best_model = built_model
            best_hyperparameters = {key: kwargs[key] for key in hyperparameters.keys()}
        else:
            print(f'Training complete taking {st-time.time()} seconds. Discarding model... ')

    return best_model, best_hyperparameters, best_score, results


def evaluate(model, X_train, X_test, y_train, y_test, plot: bool = False, title: str = 'defult'):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    scores = {'MSE_train': mean_squared_error(y_train, y_train_pred),
              'MSE_test': mean_squared_error(y_test, y_test_pred),
              'R2_train': r2_score(y_train, y_train_pred),
              'R2_test': r2_score(y_test, y_test_pred)
              }
    if plot:
        plt.scatter(y_test, y_test_pred, color='b', s=10, label='Test')
        plt.scatter(y_train, y_train_pred, color='r', s=10, label='Train')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend()
        plt.savefig(title+'_plot.svg')
    return scores


if __name__ == '__main__':
    main()
