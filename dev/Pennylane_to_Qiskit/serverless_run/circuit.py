import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


def mitarai(quantumcircuit,num_wires,paramname='x'):
    """
    parameters
    ----------
    quantumcircuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit
        
    num_wires: int
        Number of wires
        
    paramname: str
        (default:'x')
    """
    # encoding as proposed by Mitarai et al.
    num_features = num_wires
    features = ParameterVector(paramname,num_features*2)
    for i in range(num_wires):
        feature_idx = i % num_features  # Calculate the feature index using modulo
        quantumcircuit.ry(np.arcsin(features[feature_idx * 2]), i)
        quantumcircuit.rz(np.arccos(features[feature_idx * 2 + 1] ** 2), i)


def double_angle(quantumcircuit, num_wires,paramname='x'):
    """
    parameters
    ----------
    quantumcircuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit
        
    num_wires: int
        Number of wires
        
    """    
    #  creates a circuit that encodes features into wires via angle encoding with an RY then RZ gate
    #  the features are encoded 1-1 onto the qubits
    #  if more wires are passed then features the remaining wires will be filled from the beginning of the feature list
    num_features = num_wires
    features = ParameterVector(paramname,num_features*2)
    for i in range(num_wires):
        feature_index = i % num_features
        quantumcircuit.ry(features[feature_index], i)
        quantumcircuit.rz(features[feature_index], i)

def entangle_cnot(quantumcircuit,num_wires):
    """
    parameters
    ----------
    quantumcircuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit
        
    num_wires: int
        Number of wires
        
    paramname: str
        (default:'x')
    """    
    #  entangles all of the wires in a circular fashion using cnot gates
    for i in range(num_wires):
        
        if i == num_wires - 1:
            quantumcircuit.cx(i, 0)
        else:
            quantumcircuit.cx(i, i+1)


def entangle_cz(quantumcircuit,num_wires):
    """
    parameters
    ----------
    quantumcircuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit
        
    num_wires: int
        Number of wires
        
    """    
    #  entangles all of the wires in a circular fashion using cz gates
    for i in range(num_wires):
        
        if i == num_wires - 1:
            quantumcircuit.cz(i, 0)
        else:
            quantumcircuit.cz(i, i+1)


def HardwareEfficient(quantumcircuit,num_wires,paramname='theta'):
    """
    parameters
    ----------
    quantumcircuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit
        
    num_wires: int
        Number of wires
        
    paramname: str
        (default:'theta')
    """    
    parameters = ParameterVector(paramname,num_wires*3)
    for qubit in range(num_wires):
        quantumcircuit.rx(parameters[qubit * 3], qubit)  
        quantumcircuit.rz(parameters[qubit * 3 + 1], qubit)  
        quantumcircuit.rx(parameters[qubit * 3 + 2], qubit)  
    entangle_cnot(quantumcircuit,num_wires)

def circuit(nqubits,RUD=1):
    """
    parameters
    ----------
    num_wires: int
        Number of wires
        
    RUD: int
        Re-upload depth (default:1)

    returns
    -------
    qc: qiskit.circuit.quantumcircuit.QuantumCircuit
        Quantum circuit    
    """    
    qc = QuantumCircuit(nqubits)
    for i in range(RUD):
        double_angle(qc,nqubits,paramname=f'x{i}')
        qc.barrier()
        HardwareEfficient(qc,nqubits,paramname=f'theta{i}')
        qc.barrier()
    return qc