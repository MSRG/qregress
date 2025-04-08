import json
import click
import os
from .circuits.Encoders import double_angle, single_angle, iqp_embedding, mitarai, composer, entangle_cz, entangle_cnot
from .circuits.Ansatz import HardwareEfficient, EfficientSU2, TwoLocal, ModifiedPauliTwo, HadamardAnsatz

############################################
#  Lists of acceptable values
############################################


ERROR_MITIGATION_LIST = {
    'None': None,
    'MITIQ_LINEAR': 'MITIQ_LINEAR',
    'MITIQ_Richardson': 'MITIQ_Richardson',
    'M3': 'M3',
    'TREX': 'TREX'
}

POSTPROCESS_LIST = {
    'None': None,
    'simple': 'simple',
    'ridge': 'ridge',
    'lasso': 'lasso'
}

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
    'Full-CRX': TwoLocal(rot_gates=['rx', 'rz'], entangle_gates=['crx'], entanglement='complete'),
    'Modified-Pauli-CRZ': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crz', full_rotation=False),
    'Modified-Pauli-CRX': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crx', full_rotation=False),
    'Full-Pauli-CRZ': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crz', full_rotation=True),
    'Full-Pauli-CRX': ModifiedPauliTwo(rotation_block=['rx', 'rz'], entanglement='crx', full_rotation=True),
    'Hadamard': HadamardAnsatz()
}

# This is defining the grid-space of hyperparameters to search through.
hyperparameters = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'beta': [0.001, 0.01, 0.1, 1, 10]
}

