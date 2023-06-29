import joblib
import matplotlib.pyplot as plt
import click
import json
import time
from quantum.Quantum import QuantumRegressor
from quantum.circuits.Encoders import double_angle, single_angle, iqp_embedding, mitarai, composer, \
    entangle_cz, entangle_cnot
from quantum.circuits.Ansatz import HardwareEfficient, EfficientSU2, TwoLocal, ExcitationPreserving, PauliTwoDesign, \
    RealAmplitudes, HadamardAnsatz, ModifiedPauliTwo, NLocal

from sklearn.metrics import mean_squared_error, r2_score

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

ENCODER_LIST = {
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
ANSATZ_LIST = {
    'HardwareEfficient': HardwareEfficient()
}

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

    global ANSATZ
    ANSATZ = settings['ANSATZ']

    global ENCODER
    ENCODER = settings['ENCODER']


def load_dataset(file):
    print(f'Loading dataset from {file}... ')
    data = joblib.load(file)
    X = data['X']
    y = data['y']

    global X_DIM
    _, X_DIM = X.shape
    print(f'Successfully loaded {file} into X and y data. ')
    return X, y


############################################
# Main
############################################

@click.command()
@click.option('--settings', required=True, help='Settings file for running ML. ')
@click.option('--train_set', required=True, help='Datafile for training the ML model. ')
@click.option('--test_set', default=None, help='Optional datafile to use for testing and scoring the model. ')
def main(settings, train_set, test_set):
    X_train, y_train = load_dataset(train_set)
    parse_settings(settings)

    print(f'Creating and training model with given dataset {test_set} at time {time.asctime()}. ')
    st = time.time()
    model = create_model()
    model.fit(X_train, y_train, callback_interval=1)
    et = time.time()
    print(f'Training complete taking {st-et} total seconds. ')

    if test_set is not None:
        X_test, y_test = load_dataset(test_set)
        evaluate(model, X_train, X_test, y_train, y_test, plot=True)


def create_model():
    ANSATZ.set_wires(range(X_DIM))

    model = QuantumRegressor(encoder=ENCODER, variational=ANSATZ, num_qubits=X_DIM, optimizer=OPTIMIZER, device=DEVICE,
                             backend=BACKEND, postprocess=POSTPROCESS, error_mitigation=ERROR_MITIGATION)
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
