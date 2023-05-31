import pennylane as qml
from pennylane import numpy as np
import math


def mitarai(features, wires):
    for i in range(len(features)):
        qml.RY(np.arcsin(features[i]), wires=wires[i])
        qml.RZ(np.arccos(features[i]**2), wires=wires[i])


def single_angle(features, wires):
    if len(features) >> len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for j in range(math.floor(len(wires) / len(features))):
        for i in range(len(features)):
            qml.RY(features[i], wires=wires[i + j * math.floor(len(wires) / len(features))])


def double_angle(features, wires):
    if len(features) >> len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for j in range(math.floor(len(wires) / len(features))):
        for i in range(int(len(features)/2)):
            qml.RY(features[i], wires=wires[i + j * math.floor(len(wires) / len(features))])
            qml.RZ(features[i + 1], wires=wires[i + j * math.floor(len(wires) / len(features))])


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
    for i in range(len(wires)):
        qml.RX(parameters[i * len(wires)], wires=wires[i])
        qml.RZ(parameters[i * len(wires) + 1], wires=wires[i])
        qml.RX(parameters[i * len(wires) + 2], wires=wires[i])


def entangling_layers(parameters, layers, wires):
    for i in range(parameters / 3 / len(wires)):
        rotation_layer(parameters, wires)


def composer(*args):
    def new_func(features, wires):
        for arg in args:
            arg(features, wires)
    return new_func
