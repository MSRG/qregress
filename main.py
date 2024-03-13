import joblib
import click
import json
import time
import os
import itertools
import collections.abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from qiskit_ibm_provider import IBMProvider

from quantum.Quantum import QuantumRegressor
from quantum.Evaluate import evaluate
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
RE_UPLOAD_DEPTH = None
MAX_ITER = None
TOLERANCE = None
NUM_QUBITS = None


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
    # f was removed from HYPERPARAMETERS, this ensures old settings files can still run.
    if 'f' in HYPERPARAMETERS.keys():
        _ = HYPERPARAMETERS.pop('f', None)

    global RE_UPLOAD_DEPTH
    RE_UPLOAD_DEPTH = settings['RE-UPLOAD_DEPTH']

    global MAX_ITER
    MAX_ITER = settings['MAX_ITER']

    global TOLERANCE
    try:
        TOLERANCE = settings['TOLERANCE']
    except KeyError:
        TOLERANCE = None

    global NUM_QUBITS
    try:
        NUM_QUBITS = settings['NUM_QUBITS']
    except KeyError:
        NUM_QUBITS = None

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


############################################
# Main
############################################

@click.command()
@click.option('--settings', required=True, type=click.Path(exists=True), help='Settings file for running ML. ')
@click.option('--train_set', required=True, type=click.Path(exists=True), help='Datafile for training the ML model. ')
@click.option('--test_set', default=None, type=click.Path(exists=True), help='Optional datafile to use for testing '
                                                                             'and scoring the model. ')
@click.option('--scaler', required=True, type=click.Path(exists=True), help='Scaler used to unsclae y-values. ')
@click.option('--instance', default=None, help='Instance for running on IBMQ devices. ')
@click.option('--token', default=None, help='IBMQ token for running on hardware. ')
@click.option('--save_circuits', default=False, help='Whether to save a figure of encoder and ansatz circuits. ')
@click.option('--title', default=None, type=click.Path(), help='Title to use for save files. If none, infers it from '
                                                               'settings file. ')
@click.option('--resume_file', default=None, type=click.Path(exists=True), help='File to resume training from. Use '
                                                                                'the same settings file to generate '
                                                                                'the same model for training. ')
def main(settings, train_set, test_set, scaler, instance, token, save_circuits, title, resume_file):
    """
    Trains the quantum regressor with the settings in the given settings file using the dataset from the given train
    and test files. Will perform grid search on a default hyperparameter space unless they are specified. Saves scores
    and best hyperparameters to joblib dumps and graphs of performance and circuit drawings as mpl svg.
    """
    X_train, y_train = load_dataset(train_set)
    parse_settings(settings)
    if DEVICE == 'qiskit.ibmq':
        save_token(instance, token)

    global NUM_QUBITS
    global X_DIM
    if NUM_QUBITS is not None:
        X_DIM = NUM_QUBITS
    elif X_DIM == 1:  # if X_DIM is None and num_qubits wasn't specified anywhere use a default value of 2.
        NUM_QUBITS = 2
        X_DIM = NUM_QUBITS

    kwargs = create_kwargs()

    if title is None:
        title = os.path.basename(settings)
        title, _ = os.path.splitext(title)

    if save_circuits:
        plot_circuits(title)

    if test_set is not None:
        X_test, y_test = load_dataset(test_set)
    else:
        X_test, y_test = None, None

    scaler = joblib.load(scaler)

    print(f'Training model with dataset {train_set} \n at time {time.asctime()}... ')
    st = time.time()

    if len(HYPERPARAMETERS['alpha']) != 1:
        model, hyperparams, _, _ = grid_search(QuantumRegressor, HYPERPARAMETERS, X_train, y_train, **kwargs)
    else:
        model = QuantumRegressor(**kwargs)
        model.fit(X_train, y_train, load_state=resume_file)
        hyperparams = None

    et = time.time()
    print(f'Training complete taking {et - st} total seconds. ')

    # removes temporary file created during training.
    if os.path.exists(title + '_tentative_model.bin'):
        os.remove('tentative_model.bin')
    elif os.path.exists('tentative_model.bin'):
        os.remove('tentative_model.bin')

    scores, test_pred, train_pred = evaluate(model, X_train, y_train, X_test, y_test, plot=True, title=title,
                                             y_scaler=scaler)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    name = title + '_predicted_values.csv'
    train_pred, y_train, test_pred, y_test = train_pred.tolist(), y_train.tolist(), test_pred.tolist(), y_test.tolist()
    df_train = pd.DataFrame({'Predicted': train_pred, 'Reference': y_train})
    df_train['Data'] = 'Train'
    df_test = pd.DataFrame({'Predicted': test_pred, 'Reference': y_test})
    df_test['Data'] = 'Test'
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[['Data', 'Predicted', 'Reference']]

    df.to_csv(name, index=False)
    print(f'Saved predicted values as {name}')

    print(f'Model scores: {scores}. ')

    results = scores

    if len(HYPERPARAMETERS['alpha']) != 1:
        results['hyperparameters'] = hyperparams
    results_title = title + '_results.json'
    with open(results_title, 'w') as outfile:
        json.dump(results, outfile)
        pass
    print(f'Saved model results as {results_title}. ')


def plot_circuits(title):
    draw_ansatz = qml.draw_mpl(ANSATZ)
    draw_ansatz(np.random.rand(ANSATZ.num_params))
    plt.savefig(title + '_ansatz.svg')

    draw_encoder = qml.draw_mpl(ENCODER)
    draw_encoder(np.random.rand(X_DIM), range(X_DIM))
    plt.savefig(title + '_encoder.svg')


def create_kwargs():
    #  First have to apply specific ansatz settings: setting number of layers and the number of wires based on features
    ANSATZ.layers = LAYERS
    ANSATZ.set_wires(range(X_DIM))

    kwargs = {
        'encoder': ENCODER,
        'variational': ANSATZ,
        'num_qubits': X_DIM,
#       'optimizer': OPTIMIZER,
        'optimizer': "BFGS",
        'max_iterations': MAX_ITER,
        'tol': TOLERANCE,
        'device': DEVICE,
        'backend': BACKEND,
        'postprocess': POSTPROCESS,
        'error_mitigation': ERROR_MITIGATION,
        'provider': PROVIDER,
        'token': TOKEN,
        're_upload_depth': RE_UPLOAD_DEPTH,
    }
    return kwargs


def grid_search(model, hyperparameters: dict, X, y, folds: int = 5, **kwargs):
    """
    Performs a grid search on the given model. Trains the model for each combination of hyperparameters. Scores each
    model using MSE on the test fold using k-fold cross-validation saves the average across the folds as score and
    returns the best performing model with its score and hyperparameters. Any additional parameters to be passed to
    the model are handled with kwargs.

    :return: trained_model, dict: best_hyperparameters, flaot: best_score, dict: results
    """
    for x in hyperparameters.values():
        if not isinstance(x, collections.abc.Sequence):
            raise ValueError('Dictionary must contain list-like objects of values to try! ')

    kf = KFold(n_splits=folds)
    print(f'Training using {folds}-fold cross-validation. \n')

    results = {}
    best_score = float('-inf')
    best_model = None
    best_hyperparameters = {}

    param_combinations = list(itertools.product(*hyperparameters.values()))

    for combination in param_combinations:
        update = dict(zip(hyperparameters.keys(), combination))
        kwargs.update(update)
        print(f'Beginning training with hyperparameters {update}...\n')
        st = time.time()
        k_score = []
        count = 1
        for train_index, test_index in kf.split(X):
            print(f'Working on {count / folds} fold... ')
            count += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            built_model = model(**kwargs)
            built_model.fit(X_train, y_train, callback_interval=1)
            y_pred = built_model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            k_score.append(score)
        score = np.array(k_score).mean()
        results[f'{update}'] = score
        print(f'Training complete taking {time.time() - st} seconds. ')
        if score > best_score:
            print('Saving model as new best... \n')
            best_score = score
            best_model = built_model  # not sure about this line. Maybe I should return a different version of the model
            # or re-train the model on the entire set.
            best_hyperparameters = {key: kwargs[key] for key in hyperparameters.keys()}
        else:
            print('Discarding model... \n')

    with open('Grid_search.json', 'w') as outfile:
        json.dump(results, outfile)

    return best_model, best_hyperparameters, best_score, results


if __name__ == '__main__':
    main()
