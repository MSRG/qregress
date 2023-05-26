import pennylane as qml
from pennylane import numpy as np


def mitarai(features):
    for i in range(len(features)):
        qml.RY(np.arcsin(features[i]), wires=i)
        qml.RZ(np.arccos(features[i]**2), wires=i)


def single_angle(features):
    for i in range(len(features)):
        qml.RY(features[i], wires=i)


def double_angle(features):
    for i in range(len(features)):
        qml.RY(features[i], wires=i)
        qml.RZ(features[i], wires=i)


def entangle_cnot(num_qubits):
    for i in range(num_qubits):
        if i is num_qubits - 1:
            qml.CNOT(wires=(i, 0))
        else:
            qml.CNOT(wires=(i, i+1))


def entangle_cz(num_qubits):
    for i in range(num_qubits):
        if i is num_qubits - 1:
            qml.CZ(wires=(i, 0))
        else:
            qml.CZ(wires=(i, i+1))


def rotation_layer(parameters, num_qubits):
    for i in range(num_qubits):
        qml.RX(parameters[i * num_qubits], wires=i)
        qml.RZ(parameters[i * num_qubits + 1], wires=i)
        qml.RX(parameters[i * num_qubits + 2], wires=i)
