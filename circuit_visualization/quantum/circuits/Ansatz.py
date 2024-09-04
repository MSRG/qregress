import pennylane as qml
from .Encoders import entangle_cnot, entangle_cz
from qiskit.circuit.library import n_local


def rotation_layer(parameters, wires, three_rotations=True):
    if not three_rotations:
        if len(parameters) != 2 * len(wires):
            raise ValueError("Unsopported number of parameters. Expected amount should be", 3 * len(wires))
    else:
        if len(parameters) != 3 * len(wires):
            raise ValueError("Unsupported number of parameters. Expected amount should be", 3 * len(wires))
    for i in range(len(wires)):
        if three_rotations:
            qml.RX(parameters[3 * i], wires=wires[i])
            qml.RZ(parameters[3 * i + 1], wires=wires[i])
            qml.RX(parameters[3 * i + 2], wires=wires[i])
        else:
            qml.RX(parameters[2 * i], wires=wires[i])
            qml.RZ(parameters[2 * i + 1], wires=wires[i])


class HardwareEfficient:
    _entangle_types = {
        'CNOT': entangle_cnot,
        'CZ': entangle_cz
    }

    def __init__(self, wires: list = None, layers: int = 1, entangle_type: str = 'CNOT'):
        self._layers = layers
        if entangle_type not in self._entangle_types.keys():
            raise ValueError("Unexpected entangling type. Possible types are:", self._entangle_types.keys())
        self.entangler = self._entangle_types[entangle_type]
        self._wires = wires

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            self._wires = wires
        if len(parameters) != 3 * self._layers * len(self._wires):
            raise ValueError("Expected ", 3 * self._layers * len(self._wires), "parameters but got ", len(parameters))
        for i in range(self._layers):
            start_index = i * 3 * len(self._wires)
            end_index = start_index + 3 * len(self._wires)
            rotation_layer(parameters[start_index:end_index], self._wires)
            self.entangler(self._wires)

    @property
    def num_params(self):
        return 3 * self._layers * len(self._wires)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, val):
        self._layers = val

    def set_wires(self, wires):
        self._wires = wires


class EfficientSU2:

    def __init__(self,
                 wires: list = None,
                 su2_gates: list = None,
                 entanglement: str = 'linear',
                 reps: int = 1,
                 skip_final_rot: bool = False
                 ):
        if su2_gates is None:
            su2_gates = ['ry', 'rz']
        self._su2_gates = su2_gates
        self._entanglement = entanglement
        self._reps = reps
        self._skip_final_rot = skip_final_rot
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
            self._wires = wires
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, 'but received ',
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val
    

    def set_wires(self, wires):
        self._wires = wires
        # print(f"FUCKING {self._reps}")
        self._qc = n_local.EfficientSU2(num_qubits=len(self._wires), su2_gates=self._su2_gates,
                                        entanglement=self._entanglement, reps=self._reps,
                                        skip_final_rotation_layer=self._skip_final_rot)


class ExcitationPreserving:

    def __init__(self,
                 wires: list = None,
                 entanglement: str = 'linear',
                 reps: int = 1,
                 ):
        self._entanglement = entanglement
        self._reps = reps
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, 'but received ',
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val

    def set_wires(self, wires):
        self._wires = wires
        self._qc = n_local.ExcitationPreserving(num_qubits=len(self._wires), entanglement=self._entanglement,
                                                reps=self._reps)


class TwoLocal:

    def __init__(self,
                 wires: list = None,
                 entanglement: str = 'linear',
                 reps: int = 1,
                 rot_gates: list = None,
                 entangle_gates: list = None,
                 skip_final_rot: bool = True
                 ):
        self._entanglement = entanglement
        self._reps = reps
        self._rot_gates = rot_gates
        self._entangle_gates = entangle_gates
        self._skip_final_rot = skip_final_rot
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
            self._wires = wires
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, "but received ",
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val

    def set_wires(self, wires):
        self._wires = wires
        if self._entanglement == 'complete':
            entanglement = []
            for i in wires:
                for j in wires:
                    if i != j:
                        entanglement.append((i, j))
            self._entanglement = entanglement
        self._qc = n_local.TwoLocal(num_qubits=len(self._wires), entanglement=self._entanglement, reps=self._reps,
                                    rotation_blocks=self._rot_gates, entanglement_blocks=self._entangle_gates,
                                    skip_final_rotation_layer=self._skip_final_rot)


class PauliTwoDesign:

    def __init__(self, wires: list = None, reps: int = 1):
        self._reps = reps
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
            self._wires = wires
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, "but received ",
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val

    def set_wires(self, wires):
        self._wires = wires
        self._qc = n_local.PauliTwoDesign(num_qubits=len(self._wires), reps=self._reps)


class RealAmplitudes:

    def __init__(self, wires: list = None, entanglement: str = 'linear', reps: int = 1):
        self._entanglement = entanglement
        self._reps = reps
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
            self._wires = wires
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, "but received ",
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val

    def set_wires(self, wires):
        self._wires = wires
        self._qc = n_local.RealAmplitudes(num_qubits=len(self._wires), entanglement=self._entanglement, reps=self._reps)


class NLocal:

    def __init__(self,
                 wires: list = None,
                 rotation_blocks: list = None,
                 entanglement: str = None,
                 entanglement_blocks: list = None,
                 reps: int = None
                 ):
        self._rotation_blocks = rotation_blocks
        self._entanglement = entanglement
        self._entanglement_blocks = entanglement_blocks
        self._reps = reps
        self._wires = None
        self._qc = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if wires is not None:
            if len(self._wires) != len(self._wires):
                raise ValueError("Cannot override wires instance of different length")
            self._wires = wires
        qc = self._qc
        if qc.num_parameters_settable != len(parameters):
            raise ValueError("Incorrect number of parameters. Expected ", qc.num_parameters_settable, "but received ",
                             len(parameters))
        qc = qc.decompose()
        parameters = parameters.tolist()
        qc = qc.assign_parameters(parameters)
        qml_circuit = qml.from_qiskit(qc)
        qml_circuit(wires=self._wires)

    @property
    def num_params(self):
        return self._qc.num_parameters_settable

    @property
    def layers(self):
        return self._reps

    @layers.setter
    def layers(self, val):
        self._reps = val

    def set_wires(self, wires):
        self._wires = wires
        self._qc = n_local.NLocal(num_qubits=len(self._wires), rotation_blocks=self._rotation_blocks,
                                  entanglement=self._entanglement, entanglement_blocks=self._entanglement_blocks,
                                  reps=self._reps)


class ModifiedPauliTwo:

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

    def __init__(self,
                 wires: list = None,
                 entanglement: str = 'crz',
                 layers: int = 1,
                 rotation_block: list = None,
                 full_rotation: bool = True,
                 ):
        self._entangle_params = True
        if entanglement == 'cnot' or entanglement == 'cz':
            self._entangle_params = False
        if rotation_block is None:
            self._rotation_block = ['rx', 'rz']
        else:
            self._rotation_block = rotation_block
        self._entangler = self.entanglers[entanglement]
        self._layers = layers
        self._full_rotation = full_rotation
        self._wires = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters, wires: list = None):
        if len(parameters) != self.num_params:
            raise ValueError('Expected ', self.num_params, ' parameters but received ', len(parameters))
        counter = 0
        for i in range(self._layers):
            for j in range(len(self._wires)):
                self.rotations[self._rotation_block[0]](parameters[counter], j)
                counter += 1
                self.rotations[self._rotation_block[1]](parameters[counter], j)
                counter += 1
            for j in range(len(self._wires)):
                if j % 2 == 0 and j != len(self._wires) - 1:
                    if self._entangle_params:
                        self._entangler(parameters[counter], (j + 1, j))
                        counter += 1
                    else:
                        self._entangler((j + 1, j))

            for j in range(len(self._wires)):
                if self._full_rotation:
                    self.rotations[self._rotation_block[0]](parameters[counter], j)
                    counter += 1
                    self.rotations[self._rotation_block[1]](parameters[counter], j)
                    counter += 1
                elif not self._full_rotation:
                    if j != 0 and j != len(self._wires) - 1:
                        self.rotations[self._rotation_block[0]](parameters[counter], j)
                        counter += 1
                        self.rotations[self._rotation_block[1]](parameters[counter], j)
            for j in range(len(self._wires)):
                if j % 2 != 0 and j != len(self._wires) - 1:
                    if self._entangle_params:
                        self._entangler(parameters[counter], (j + 1, j))
                        counter += 1
                    else:
                        self._entangler((j + 1, j))

    @property
    def num_params(self):
        num = 0
        if self._full_rotation:
            if self._entangle_params:
                num = self._layers * (4 * len(self._wires) + len(self._wires) - 1)
            elif not self._entangle_params:
                num = self._layers * (4 * len(self._wires))
        elif not self._full_rotation:
            if self._entangle_params:
                num = self._layers * (4 * len(self._wires) + len(self._wires) - 5)
            elif not self._entangle_params:
                num = self._layers * (4 * len(self._wires) - 4)
        return num

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, val):
        self._layers = val

    def set_wires(self, wires):
        self._wires = wires


class HadamardAnsatz:

    def __init__(self, wires: list = None, layers: int = 1):
        self._layers = layers
        self._wires = None
        if wires is not None:
            self.set_wires(wires)

    def __call__(self, parameters: list, wires: list = None):
        if wires is not None:
            self._wires = wires
        counter = 0
        for _ in range(self._layers):
            for i in range(len(self._wires)):
                qml.Hadamard(wires=self._wires[i])
            for i in range(len(self._wires)):
                if i != len(self._wires) - 1:
                    qml.CZ((i, i + 1))
                qml.RX(parameters[counter], wires=self._wires[i])
                counter += 1

    @property
    def num_params(self):
        return self._layers * len(self._wires)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, val):
        self._layers = val

    def set_wires(self, wires):
        self._wires = wires
