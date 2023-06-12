import pennylane as qml
from pennylane import numpy as np


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


def composer(*args):
    def new_func(features, wires):
        for arg in args:
            num_params = arg.__code__.co_argcount
            if num_params == 1:
                arg(wires)
            else:
                arg(features, wires)
    return new_func


def amplitude_embedding(features, wires, pad_with=None):
    if len(features) != 2 ** len(wires) and pad_with is None:
        raise ValueError('Should be encoding 2^n features into n qubits. If you want to encode fewer features specify '
                         'a padding')
    if pad_with == 'self' and len(features) != 2 ** len(wires):
        diff = 2 ** len(wires) - len(features)
        features = np.concatenate([features, np.tile(features, diff // len(features) + 1)])[:2 ** len(wires)]
        pad_with = None
    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=pad_with, normalize=True)

