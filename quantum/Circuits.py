import pennylane as qml
from pennylane import numpy as np
import math


def mitarai(features, wires):
    for i in range(len(features)):
        qml.RY(np.arcsin(features[i]), wires=wires[i])
        qml.RZ(np.arccos(features[i]**2), wires=wires[i])


def single_angle(features, wires):
    if len(features) > len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for i in range(len(wires)):
        feature_index = i % len(features)
        qml.RY(features[feature_index], wires=wires[i])


def double_angle(features, wires):
    if len(features) >> len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for i in range(len(wires)):
        feature_index = i % len(features)
        qml.RY(features[feature_index], wires=wires[i])
        qml.RZ(features[feature_index], wires=wires[i])


def entangle_cnot(wires):
    for i in wires:
        if i == len(wires) - 1:
            qml.CNOT(wires=(i, 0))
        else:
            qml.CNOT(wires=(i, i+1))


def entangle_cz(wires):
    for i in wires:
        if i == len(wires) - 1:
            qml.CZ(wires=(i, 0))
        else:
            qml.CZ(wires=(i, i+1))


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


def composer(*args):
    def new_func(features, wires):
        for arg in args:
            num_params = arg.__code__.co_argcount
            if num_params == 1:
                arg(wires)
            else:
                arg(features, wires)
    return new_func
