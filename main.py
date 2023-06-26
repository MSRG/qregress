import joblib
import matplotlib.pyplot as plt
import numpy as np
import click
import json
import pandas
from quantum.Quantum import QuantumRegressor
from quantum.circuits.Encoders import double_angle, single_angle, iqp_embedding, mitarai, composer, \
    entangle_cz, entangle_cnot
from quantum.circuits.Ansatz import HardwareEfficient, EfficientSU2, TwoLocal, ExcitationPreserving, PauliTwoDesign, \
    RealAmplitudes, HadamardAnsatz, ModifiedPauliTwo, NLocal

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Global variables
OPTIMIZER = None
SHOTS = None
LOAD_FILE = None
LENGTH = None
X_DIM = None
BACKEND = None
DEVICE = None
SCALE_FACTORS = None


ERROR_MITIGATION = [
    'MITIQ_LINEAR',
    'MITIQ_Richardson',
    'M3',
    'TREX'
]


ENCODERS = {
    'M': mitarai,
    'A1': single_angle,
    'A2': double_angle,
    'IQP': iqp_embedding,
    'M-M-CNOT': composer(mitarai, entangle_cnot, mitarai, entangle_cnot),
    'A1-A1-CNOT': composer(single_angle, entangle_cnot, single_angle, entangle_cnot),
    'A2-A2-CNOT': composer(double_angle, entangle_cnot, double_angle, entangle_cnot),
    'M-A1-CNOT': composer(mitarai, entangle_cnot, single_angle, entangle_cnot),
    'M-A2-CNOT': composer(mitarai, entangle_cnot, double_angle, entangle_cnot),
    'M-M-CZ': composer(mitarai, entangle_cz, mitarai, entangle_cz),
    'A1-A1-CZ': composer(single_angle, entangle_cz, single_angle, entangle_cz),
    'A2-A2-CZ': composer(double_angle, entangle_cz, double_angle, entangle_cz),
    'M-A1-CZ': composer(mitarai, entangle_cz, single_angle, entangle_cz),
    'M-A2-CZ': composer(mitarai, entangle_cz, double_angle, entangle_cz),
}

# TODO: Create a full list of ansatz to be used in the experiment
ANSATZES = {
    'HardwareEfficient': None
}

MEASUREMENTS = [None, 'simple', 'ridge', 'lasso']


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

    global LOAD_FILE
    LOAD_FILE = settings['LOAD_FILE']

    global LENGTH
    LENGTH = settings['LENGTH']

    global X_DIM
    X_DIM = settings['X_DIM']

    global BACKEND
    BACKEND = settings['BACKEND']

    global DEVICE
    DEVICE = settings['DEVICE']

    global SCALE_FACTORS
    SCALE_FACTORS = settings['SCALE_FACTORS']

############################################
# Dataset preparation
############################################


def load_file(y_label: str = 'BSE'):
    length = LENGTH
    df = joblib.load(LOAD_FILE)
    if length is None:
        length = len(df[y_label])
    y = df[y_label][:length]
    y = np.array(y)
    x = df.drop([y_label], axis=1)[:length]
    return x, y


def split(x, y, test_ratio: float = 0.2, x_dim: int = None):
    if x_dim is not None:
        raise NotImplementedError('PCA / component analysis is not yet implemented.')
    scaler = MinMaxScaler
    x_scaler = scaler((-1, 1))
    y_scaler = scaler((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)
    sc_X_tr = x_scaler.fit_transform(X_train)
    sc_X_te = x_scaler.transform(X_test)
    sc_y_tr = y_scaler.fit_transform(y_train).reshape(-1)
    sc_y_te = y_scaler.transform(y_test).reshape(-1)

    return sc_X_tr, sc_y_tr, sc_X_te, sc_y_te  # returns (X_train y_train X_test y_test)


############################################
# Main
############################################

@click.command()
@click.option('--test_ratio', default=0.2, help='Ratio of data to use for testing')
@click.option('--x_dim', default=16, help='Number of features to reduce X dataset to')
@click.option('--ansatz', default=None, help='Optionally specify a specific ansatz to run')
@click.option('--encoder', default=None, help='Optionally specify a specific encoder circuit to run')
@click.option('--measurement', default=None, help='Optionally specify a specific classical postprocessing')
@click.option('--error_mitigation', default=None, help='Specify error mitigation to be used, defaults to None')
@click.option('--settings', required=True, help='Settings file for running ML')
def main(test_ratio, x_dim, new_ansatz, new_encoder, measurement, error_mitigation, settings):

    parse_settings(settings_file=settings)
    X, y = load_file()
    X_train, y_train, X_test, y_test = split(X, y, test_ratio, x_dim)
    num_qubits = x_dim

    if new_encoder is None:
        encoders = ENCODERS
    else:
        encoders = {new_encoder, ENCODERS[new_encoder]}
    if new_ansatz is None:
        ansatzes = ANSATZES
    else:
        ansatzes = {new_ansatz, ANSATZES[new_ansatz]}
    if measurement is None:
        measurements = MEASUREMENTS
    else:
        if measurement == 'None':
            measurements = [None]
        else:
            measurements = measurement

    scores = {}
    for encoder_name, encoder in encoders.values():
        for ansatz_name, ansatz in ansatzes.values():
            for measurement in measurements:
                model = create_model(encoder, ansatz, measurement, num_qubits, error_mitigation)
                model.fit(X_train, y_train, callback_interval=1)
                score = evaluate(model, X_train, X_test, y_train, y_test, plot=True, title=encoder_name + ansatz_name)
                scores[encoder_name+'_'+ansatz_name+'_'+measurement] = score
    return scores


def create_model(encoder, variational, measurement, num_qubits, error_mitigation):
    if error_mitigation not in ERROR_MITIGATION:
        raise ValueError(f'Unexpected type for error mitigation. Type must be in {ERROR_MITIGATION}')

    model = QuantumRegressor(encoder, variational, num_qubits=num_qubits, optimizer=OPTIMIZER, device=DEVICE,
                             backend=BACKEND, postprocess=measurement, error_mitigation=error_mitigation)
    return model


def evaluate(model, X_train, X_test, y_train, y_test, plot: bool = False, title: str = 'defult'):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    scores = {'MSE_train': mean_squared_error(y_train, y_train_pred),
              'MSE_test': mean_squared_error(y_test, y_test_pred),
              'R2_train': r2_score(y_train, y_train_pred),
              'R2_test': r2_score(y_test, y_test_pred)
              }
    if plot:
        # TODO: make plotting better and save them with nice titles. Add axis, etc.
        plt.title(title)
        plt.scatter(y_test, y_test_pred, color='b', s=10, label='Test')
        plt.scatter(y_train, y_train_pred, color='r', s=10, label='Train')
        plt.legend()
    return scores


if __name__ == '__main__':
    main()
