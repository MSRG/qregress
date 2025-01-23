from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator

def transpile_remote(initial_circuit,backend,optimization_level=0):
    '''
    Remote transpilation based on https://docs.quantum.ibm.com/guides/serverless-first-program

    parameters
    ----------
    initial_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Unmapped circuit
        
    optimization_level: int
        Circuit optimization level
    
    backend: qiskit_ibm_runtime.ibm_backend.IBMBackend

        Backend can be real (e.g. qiskit_ibm_runtime.ibm_backend.IBMBackend) or fake (e.g. qiskit_ibm_runtime.fake_provider.backends.quebec.fake_quebec.FakeQuebec)

    returns
    -------
    qc: qiskit.circuit.quantumcircuit.QuantumCircuit
        Mapped circuit

    
    mapped_observables: list
        Contains the desired mapped observables 
    
    '''            
    
    num_qubits = initial_circuit.num_qubits
    
    # Generate pass manager
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
    qc = pm.run(initial_circuit)

    # Observables
    observables_labels = ''.join(['I']*(num_qubits-1))+"Z"
    observables = [SparsePauliOp(observables_labels)]
    mapped_observables = [observable.apply_layout(qc.layout) for observable in observables]
    
    return qc, mapped_observables