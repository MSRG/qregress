import pennylane as qml
from Encoders import entangle_cnot, entangle_cz
from qiskit.circuit.library.n_local import EfficientSU2, ExcitationPreserving
from typing import Union


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


def efficient_su2(
        parameters,
        wires: Union[list, int],
        su2_gates=None,
        entanglement="linear",
        reps: int = 1):
    #  Implements Qiskit's EfficientSU2 ansatz template by converting it using the qiskit-pennylane plugin
    if su2_gates is None:
        su2_gates = ['ry', 'rz']
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = EfficientSU2(num_qubits=num_qubits, su2_gates=su2_gates, entanglement=entanglement, reps=reps)
    if qc.num_parameters_settable != len(parameters):
        raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, ' but received ',
                         len(parameters))
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)


def excitation_preserving(
        parameters,
        wires: Union[list, int],
        entanglement='linear',
        reps=1):
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = ExcitationPreserving(num_qubits=num_qubits, entanglement=entanglement, reps=reps)
    if qc.num_parameters_settable != len(parameters):
        raise ValueError('Incorrect number of parameters. Expected ', qc.num_parameters_settable, ' but received ',
                         len(parameters))
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)
