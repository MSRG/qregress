import json
import click
from quantum.circuits.Encoders import double_angle, single_angle, iqp_embedding, mitarai, composer, \
    entangle_cz, entangle_cnot
from quantum.circuits.Ansatz import HardwareEfficient, EfficientSU2, TwoLocal


# Global variables
OPTIMIZER = None
SHOTS = None
LENGTH = None
BACKEND = None
DEVICE = None
SCALE_FACTORS = None

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
    'HWE_CNOT': HardwareEfficient(),
    'HWE_CZ': HardwareEfficient(entangle_type='CZ'),
    'ESU2': EfficientSU2(skip_final_rot=True),
    'Efficient_CRZ': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='linear'),
    'Efficient_CRX': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crx'], entanglement='linear'),
    'Full_CRZ': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='complete'),
    'Full_CRX': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crz'], entanglement='complete'),

}

POSTPROCESS = {
    'None': None,
    'simple': 'simple',
    'ridge': 'ridge',
    'lasso': 'lasso'
}


def create_settings(filename: str, settings: dict, postprocess, error_mitigation, shots, backend, device,
                    optimizer, layers, scale_factors=None):
    """
    Takes inputs for all of the settings to be used in the QML model and creates a dictionary of the corresponding
    settings. Then is dumped into JSON and saved as filename.json. Filename parameter should not include the extension.

    """
    settings['POSTPROCESS'] = postprocess
    settings['ERROR_MITIGATION'] = error_mitigation
    settings['SHOTS'] = shots
    settings['DEVICE'] = device
    settings['BACKEND'] = backend
    settings['OPTIMIZER'] = optimizer
    settings['SCALE_FACTORS'] = scale_factors
    settings['LAYERS'] = layers

    filename = filename + '.json'

    with open(filename, 'w') as outfile:
        json.dump(settings, outfile)


def create_combinations(encoder: str = None, ansatz: str = None, **kwargs):
    """
    Creates combinations of every ansatz with every available encoder. Alternativaley, an encoder or ansatz can be
    specified then it will create combinations of that ansatz/encoder with every other. If both an encoder and
    ansatz are specified then it will create a dictionary with a generic name for the combination.

    :return:
    """
    if encoder is None:
        encoder = ENCODERS.keys()
    else:
        encoder = [encoder]
    if ansatz is None:
        ansatz = ANSATZES.keys()
    else:
        ansatz = [ansatz]

    my_dict = {}
    for encoder_val in encoder:
        for ansatz_val in ansatz:
            settings = {
                'ANSATZ': ansatz_val,
                'ENCODERS': encoder_val
            }
            filename = f'{encoder_val}_{ansatz_val}'
            my_dict[filename] = settings
    return my_dict


@click.command()
@click.option('--encoder', default=None, help='Encoder circuit to generate settings for. ')
@click.option('--ansatz', default=None, help='Ansatz circuit to generate settings for. ')
@click.option('--layers', default=None, help='Layers to use for ansatz. ')
@click.option('--device', default='qulacs.simulator', help='Device to run on. ')
@click.option('--backend', default=None, help='If running on IBMQ device, specify a backend here. ')
@click.option('--shots', default=None, help='Number of shots to estimate expectation values from. If none is '
                                            'specified will use the device default. ')
@click.option('--optimizer', default=None, help='Specify an optimizer for the model. COBYLA is recommended for '
                                                'noiseless and SPSA or Nelder-Mead for noisy. ')
@click.option('--error_mitigation', default=None, help='Specify an error mitigation method if using a noisy device. '
                                                       'Leave blank for none. ')
@click.option('--post_process', default=None, help='Specify a post-processing type. Leave blank for none. ')
@click.option('--file_name', default=None, help='Name for the file to be saved as. Only specify if creating a single '
                                                'settings file. ')
def main(encoder, ansatz, layers, device, backend, shots, optimizer, error_mitigation, post_process, file_name):
    settings = create_combinations(encoder, ansatz)
    if file_name is not None:
        new_settings = {}
        for val in settings:
            new_settings[file_name] = val
        settings = new_settings

    for name, setting in settings.values():
        create_settings(filename=name, settings=setting, postprocess=post_process, error_mitigation=error_mitigation,
                        shots=shots, backend=backend, device=device, optimizer=optimizer, layers=layers)


if __name__ == '__main__':
    main()
