import pennylane as qml
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


class QuantumRegressor:
    credentials = None

    def __init__(
            self,
            encoder,
            variational,
            num_qubits,
            optimizer='COBYLA',
            device='default.qubit'):
        self._set_device(device, num_qubits)
        self.x = None
        self.y = None
        self.params = None
        self.encoder = encoder
        self.variational = variational
        self.num_qubits = num_qubits
        self.qnode = qml.QNode(self._circuit, self.device)

    def _set_device(self, device, num_qubits):
        if device == 'qiskit.ibmq' and QuantumRegressor.credentials is None:
            QuantumRegressor.credentials = []
            QuantumRegressor.credentials.append(input('Please input API token'))
            QuantumRegressor.credentials.append(input('Please input backend'))
        self.device = qml.device(device, wires=num_qubits, backend=QuantumRegressor.credentials[1], ibmqx_token=QuantumRegressor.credentials[0])

    def _circuit(self, features, parameters):
        self.encoder(features, wires=range(self.num_qubits))
        self.variational(parameters, wires=range(self.num_qubits))
        return qml.expval(qml.PauliZ(0))

    def _cost(self, parameters):
        predicted_y = [self.qnode(x, parameters) for x in self.x]
        return mean_squared_error(self.y, predicted_y)

    def fit(self, x, y, initial_parameters=None, detailed_results=False):
        if initial_parameters is None:
            raise ValueError('Missing initial parameters')  # to do, assign a random set of initial parameters if
            # none are passed
        self.x = x
        self.y = y
        params = initial_parameters
        opt_result = minimize(self._cost, x0=params, method='COBYLA')
        self.params = opt_result['x']
        if detailed_results:
            return opt_result
        return self.params

    def predict(self, x):
        if self.params is None:
            raise ValueError('Model must be trained first!')
        return [self.qnode(features=features, parameters=self.params) for features in x]
