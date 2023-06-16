import pennylane as qml
from pennylane import numpy as np


def mitarai(features, wires):
    #  encoding as proposed by Mitarai et al.
    for i in range(len(features)):
        qml.RY(np.arcsin(features[i]), wires=wires[i])
        qml.RZ(np.arccos(features[i]**2), wires=wires[i])


def single_angle(features, wires):
    #  creates a circuit that encodes features into wires via angle encoding with a single RY gate
    #  the features are encoded 1-1 onto the qubits
    #  if more wires are passed then features the remaining wires will be filled from the beginning of the feature list
    if len(features) > len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for i in range(len(wires)):
        feature_index = i % len(features)
        qml.RY(features[feature_index], wires=wires[i])


def double_angle(features, wires):
    #  creates a circuit that encodes features into wires via angle encoding with an RY then RZ gate
    #  the features are encoded 1-1 onto the qubits
    #  if more wires are passed then features the remaining wires will be filled from the beginning of the feature list
    if len(features) >> len(wires):
        raise ValueError("Cannot encode more features than there are wires")
    for i in range(len(wires)):
        feature_index = i % len(features)
        qml.RY(features[feature_index], wires=wires[i])
        qml.RZ(features[feature_index], wires=wires[i])


def entangle_cnot(wires):
    #  entangles all of the wires in a circular fashion using cnot gates
    for i in wires:
        if i == len(wires) - 1:
            qml.CNOT(wires=(i, 0))
        else:
            qml.CNOT(wires=(i, i+1))


def entangle_cz(wires):
    #  entangles all of the wires in a circular fashion using cz gates
    for i in wires:
        if i == len(wires) - 1:
            qml.CZ(wires=(i, 0))
        else:
            qml.CZ(wires=(i, i+1))


def composer(*args):
    #  utility function used to compose encoding functions together to achieve layers of encoding and entangling
    #  checks the number of parameters of the input, if it takes only one 1 parameter it is assumed to be wires
    #  returns the new function that executes the input functions in given order
    def new_func(features, wires):
        for arg in args:
            num_params = arg.__code__.co_argcount
            if num_params == 1:
                arg(wires)
            else:
                arg(features, wires)
    return new_func


def iqp_embedding(features, wires, layers=1):
    if len(features) >> len(wires):
        raise ValueError('Cannot encode more features than wires')
    if len(features) << len(wires):
        qml.IQPEmbedding(features, wires[:len(features)], layers)
        qml.IQPEmbedding(features[:len(wires)-len(features)], wires[len(features):], layers)
    else:
        qml.IQPEmbedding(features, wires, layers)


def displacement_embedding(features, wires):
    if len(features) >> len(wires):
        raise ValueError('Cannot encode more features than wires')
    if len(features) << len(wires):
        qml.DisplacementEmbedding(features, wires[:len(features)])
        qml.DisplacementEmbedding(features[:len(wires)-len(features)], wires[len(features):])
    else:
        qml.DisplacementEmbedding(features, wires)


def amplitude_embedding(features, wires, pad_with=None):
    if len(features) != 2 ** len(wires) and pad_with is None:
        raise ValueError('Should be encoding 2^n features into n qubits. If you want to encode fewer features specify '
                         'a padding')
    if pad_with == 'self' and len(features) != 2 ** len(wires):
        diff = 2 ** len(wires) - len(features)
        features = np.concatenate([features, np.tile(features, diff // len(features) + 1)])[:2 ** len(wires)]
        pad_with = None
    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=pad_with, normalize=True)

