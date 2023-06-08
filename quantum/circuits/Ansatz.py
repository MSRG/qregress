import pennylane as qml
from pennylane import numpy as np
from Encoders import entangle_cnot, entangle_cz


def rotation_layer(parameters, wires):
    if len(parameters) != 3 * len(wires):
        raise ValueError("Unsupported number of parameters. Expected amount should be 3 * len(wires)")
    for i in range(len(wires)):
        qml.RX(parameters[3 * i], wires=wires[i])
        qml.RZ(parameters[3 * i + 1], wires=wires[i])
        qml.RX(parameters[3 * i + 2], wires=wires[i])


def entangling_layers(parameters, wires, entangle_type='CNOT'):
    entanglers = {
        'CNOT': entangle_cnot,
        'CZ': entangle_cz
    }
    if len(parameters) % (3 * len(wires)) != 0:
        raise ValueError("Unsupported number of parameters. Expected amount should be 3 * layers * wires")
    if entanglers[entangle_type] is None:
        raise ValueError("Unexpected entangling type")
    layers = len(parameters) // (3 * len(wires))
    for i in range(layers):
        start_index = i * 3 * len(wires)
        end_index = start_index + 3 * len(wires)
        rotation_layer(parameters[start_index:end_index], wires)
        entanglers[entangle_type](wires)
