import sys
# !{sys.executable} -m pip install "git+https://github.com/PennyLaneAI/pennylane-qiskit.git#egg=pennylane-qiskit" --force-reinstall
import pennylane as qml
from pennylane import numpy as pnp
from qiskit_ibm_runtime import QiskitRuntimeService
from pennylane_qiskit.qiskit_device import qiskit_session
# QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
# To access saved credentials for the IBM quantum channel and select an instance
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend(name= 'ibm_quebec')
dev = qml.device(
    "qiskit.remote",
    wires=127,
    backend=backend,
    resilience_level=1,
    optimization_level=1,
    seed_transpiler=42
)

with qiskit_session(dev,max_time='2h') as session:
    print(session.details())

    # Add your code here
    
    @qml.qnode(dev)
    def circuit(theta):
      qml.RX(theta,wires=0)
      return qml.expval(qml.PauliZ(0))
    
    opt = qml.GradientDescentOptimizer(stepsize=0.3)
    
    theta = pnp.array(0.5)
    for i in range(10):
      theta = opt.step(circuit,theta)
      print('theta: ', theta, 'circuit: ', circuit(theta))
