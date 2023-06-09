import pennylane as qml
from Encoders import entangle_cnot, entangle_cz
from qiskit.circuit.library.n_local import EfficientSU2, ExcitationPreserving, TwoLocal
from qiskit.circuit.library.n_local import PauliTwoDesign, RealAmplitudes, NLocal
from typing import Union


def rotation_layer(parameters, wires, three_rotations=True):
    if len(parameters) != 3 * len(wires):
        raise ValueError("Unsupported number of parameters. Expected amount should be 3 * len(wires)")
    for i in range(len(wires)):
        qml.RX(parameters[3 * i], wires=wires[i])
        qml.RZ(parameters[3 * i + 1], wires=wires[i])
        if three_rotations:
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


def two_local(
        parameters,
        wires: Union[list, int],
        entanglement='linear',
        reps=1):
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = TwoLocal(num_qubits=num_qubits, entanglement=entanglement, reps=reps)
    if qc.num_parameters_settable != len(parameters):
        raise ValueError('Incorrect number of parameters. Expected ', qc.num_parameters_settable, ' but received ',
                         len(parameters))
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)


def pauli_two_design(
        parameters,
        wires: Union[list, int, tuple],
        reps=1):
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = PauliTwoDesign(num_qubits=num_qubits, reps=reps)
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)


def real_amplitudes(
        parameters,
        wires: Union[list, int, tuple],
        entanglement='linear',
        reps=1):
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = RealAmplitudes(num_qubits=num_qubits, entanglement=entanglement, reps=reps)
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)


def n_local(
        parameters,
        wires: Union[list, int, tuple],
        rotation_blocks=None,
        entanglement=None,
        reps=1):
    if type(wires) is list or tuple:
        num_qubits = len(wires)
    else:
        num_qubits = wires
    qc = NLocal(num_qubits=num_qubits, rotation_blocks=rotation_blocks, entanglement=entanglement, reps=reps)
    qc = qc.decompose()
    qc = qc.bind_parameters(parameters)
    qml_circuit = qml.from_qiskit(qc)
    qml_circuit(wires=wires)


def modified_pauli_two(parameters: list, wires: list, entanglement: str = 'crz', layers: int = 1,
                       rotation_block: list = None, full_rotation: bool = True):
    entanglers = {
        'crz': qml.CRZ,
        'crx': qml.CRX,
        'cnot': qml.CNOT,
        'cz': qml.CZ
    }
    rotations = {
        'rz': qml.RZ,
        'rx': qml.RX,
        'ry': qml.RY
    }
    entangle_param = True
    if entanglement == 'cnot' or entanglement == 'cz':
        entangle_param = False
    if full_rotation:
        if entangle_param:
            if len(parameters) != layers * (4 * len(wires) + len(wires) - 1):
                raise ValueError('Expected ', layers * (4 * len(wires) + len(wires) - 1),
                                 'parameters, but got: ', len(parameters))
        elif not entangle_param:
            if len(parameters) != layers * (4 * len(wires)):
                raise ValueError('Expected ', layers * (4 * len(wires)),
                                 'parameters, but got: ', len(parameters))
    elif not full_rotation:
        if entangle_param:
            if len(parameters) != layers * (4 * len(wires) + len(wires) - 3):
                raise ValueError('Expected ', layers * (4 * len(wires) + len(wires) - 3),
                                 'parameters, but got: ', len(parameters))
        elif not entangle_param:
            if len(parameters) != layers * (4 * len(wires) - 2):
                raise ValueError('Expected ', layers * (4 * len(wires) - 2),
                                 'parameters, but got: ', len(parameters))
    if rotation_block is None:
        rotation_block = ['rx', 'rz']
    counter = 0
    entangler = entanglers[entanglement]
    for i in range(layers):
        for j in range(len(wires)):
            rotations[rotation_block[0]](parameters[counter], j)
            counter += 1
            rotations[rotation_block[1]](parameters[counter], j)
            counter += 1
        for j in range(len(wires)):
            if j % 2 == 0:
                if entangle_param:
                    entangler(parameters[counter], (j, j+1))
                    counter += 1
                else:
                    entangler((j, j+1))
        for j in range(len(wires)):
            if full_rotation is True:
                rotations[rotation_block[0]](parameters[counter], j)
                counter += 1
                rotations[rotation_block[1]](parameters[counter], j)
                counter += 1
            elif full_rotation is not True:
                if j != 0 or j != len(wires) - 1:
                    rotations[rotation_block[0]](parameters[counter], j)
                    counter += 1
                    rotations[rotation_block[1]](parameters[counter], j)
        for j in range(len(wires)):
            if j % 2 != 0 and j != len(wires) - 1:
                if entangle_param:
                    entangler(parameters[counter], (j, j+1))
                    counter += 1
                else:
                    entangler((j, j+1))


def hadamard_ansatz(parameters: list, wires: list, layers: int = 1):
    counter = 0
    for _ in range(layers):
        for i in range(len(wires)):
            qml.Hadamard(wires=wires[i])
        for i in range(len(wires)):
            if i != len(wires) - 1:
                qml.CZ((i, i+1))
            qml.RX(parameters[counter], wires=wires[i])
            counter += 1
