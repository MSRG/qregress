import json
import click
import numpy as np
from quantum.circuits.Encoders import double_angle, single_angle, iqp_embedding, mitarai, composer, \
    entangle_cz, entangle_cnot
from quantum.circuits.Ansatz import HardwareEfficient, EfficientSU2, TwoLocal, ModifiedPauliTwo, HadamardAnsatz


ERROR_MITIGATION = [
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
    'HWE-CNOT': HardwareEfficient(),
    'HWE-CZ': HardwareEfficient(entangle_type='CZ'),
    'ESU2': EfficientSU2(skip_final_rot=True),
    'Efficient-CRZ': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='linear'),
    'Efficient-CRX': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crx'], entanglement='linear'),
    'Full-CRZ': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='complete'),
    'Full-CRX': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='complete'),
    'Modified-Pauli-CRZ': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crz', full_rotation=False),
    'Modified-Pauli-CRX': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crx', full_rotation=False),
    'Full-Pauli-CRZ': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crz', full_rotation=True),
    'Full-Pauli-CRX': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crx', full_rotation=True),
    'Hadamard': HadamardAnsatz()
}

POSTPROCESS = {
    'None': None,
    'simple': 'simple',
    'ridge': 'ridge',
    'lasso': 'lasso'
}

# This is defining the grid-space of hyperparameters to search through.
hyperparameters = {
    'f': np.arange(0.5, 2.5, 0.2).tolist(),
    'alpha': np.arange(0, 1, 0.05).tolist(),
    'beta': np.arange(0, 1, 0.05).tolist()
}


def create_settings(filename: str, settings: dict, postprocess, error_mitigation, shots, backend, device,
                    optimizer, layers, scale_factors=None):
    """
    Takes inputs for all of the settings to be used in the QML model and creates a dictionary of the corresponding
    settings. Then is dumped into JSON and saved as filename.json. Filename parameter should not include the extension.

    """
    print(f'Creating settings file {filename}...')
    # check post-process, if None then reduce hyperparameter search space to decrease redundancies and
    # improve train time. Alpha and beta parameters are not used when post-process is None so are set to [0] as to not
    # iterate over the same settings during grid-search.
    if postprocess is None:
        hyperparameters['alpha'] = [0]
        hyperparameters['beta'] = [0]
    settings['POSTPROCESS'] = postprocess
    settings['ERROR_MITIGATION'] = error_mitigation
    settings['SHOTS'] = shots
    settings['DEVICE'] = device
    settings['BACKEND'] = backend
    settings['OPTIMIZER'] = optimizer
    settings['SCALE_FACTORS'] = scale_factors
    settings['LAYERS'] = layers
    settings['HYPERPARAMETERS'] = hyperparameters

    filename = filename + '.json'

    with open(filename, 'w') as outfile:
        json.dump(settings, outfile)
    print(f'Successfully created {filename}. ')


def create_combinations(encoder: str = None, ansatz: str = None):
    """
    Creates combinations of every ansatz with every available encoder. Alternativaley, an encoder or ansatz can be
    specified then it will create combinations of that ansatz/encoder with every other. If both an encoder and
    ansatz are specified then it will create a dictionary with a generic name for the combination.

    :return:
    """
    if encoder is None:
        encoder = ENCODER_LIST.keys()
    else:
        encoder = [encoder]
    if ansatz is None:
        ansatz = ANSATZ_LIST.keys()
    else:
        ansatz = [ansatz]

    my_dict = {}
    for encoder_val in encoder:
        for ansatz_val in ansatz:
            settings = {
                'ANSATZ': ansatz_val,
                'ENCODER': encoder_val
            }
            filename = f'{encoder_val}_{ansatz_val}'
            my_dict[filename] = settings
    return my_dict


@click.command()
@click.option('--encoder', default=None, help='Encoder circuit to generate settings for. ')
@click.option('--ansatz', default=None, help='Ansatz circuit to generate settings for. ')
@click.option('--layers', default=1, help='Layers to use for ansatz. ')
@click.option('--device', default='qulacs.simulator', help='Device to run on. ')
@click.option('--backend', default=None, help='If running on IBMQ device, specify a backend here. ')
@click.option('--shots', default=None, help='Number of shots to estimate expectation values from. If none is '
                                            'specified will use the device default. ')
@click.option('--optimizer', required=True, help='Specify an optimizer for the model. COBYLA is recommended for '
                                                 'noiseless and SPSA or Nelder-Mead for noisy. ')
@click.option('--error_mitigation', default=None, help='Specify an error mitigation method if using a noisy device. '
                                                       'Leave blank for none. ')
@click.option('--post_process', default=None, help='Specify a post-processing type. Leave blank for none. ')
@click.option('--file_name', default=None, type=click.Path(), help='Name for the file to be saved as. Only specify if '
                                                                   'creating a single settings file. ')
def main(encoder, ansatz, layers, device, backend, shots, optimizer, error_mitigation, post_process, file_name):
    settings = create_combinations(encoder, ansatz)
    if file_name is not None:
        new_settings = {}
        for _, val in settings.items():
            new_settings[file_name] = val
        settings = new_settings
    try:
        if shots is not None:
            shots = int(shots)
    except ValueError:
        shots = None
        print("Could not read shots, ensure it's interpretable as int. Proceeding with device default. ")

    for name, setting in settings.items():
        create_settings(filename=name, settings=setting, postprocess=post_process, error_mitigation=error_mitigation,
                        shots=shots, backend=backend, device=device, optimizer=optimizer, layers=layers)


if __name__ == '__main__':
    main()
