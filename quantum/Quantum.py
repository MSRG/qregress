import pennylane as qml
from sklearn.metrics import mean_squared_error


class QuantumRegressor:
    def __init__(self, encoder, variational, num_qubits, device='default.qubit',
                max_iterations=100):
        self.x = None
        self.y = None
        self.params = None
        self.encoder = encoder
        self.variational = variational
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits)
        self.max_iterations = max_iterations
        self.optimizer = qml.SPSAOptimizer(maxiter=max_iterations)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, features, parameters):
        self.encoder(features, wires=range(self.num_qubits))
        self.variational(parameters, wires=range(self.num_qubits))
        return qml.expval(qml.PauliZ(0))

    def _cost(self, parameters):
        predicted_y = [self.qnode(x, parameters) for x in self.x]
        return mean_squared_error(self.y, predicted_y)

    def fit(self, x, y, initial_parameters=None):
        if initial_parameters is None:
            pass
        self.x = x
        self.y = y
        params = initial_parameters
        cst = []
        for _ in range(self.max_iterations):
            params, cost = self.optimizer.step_and_cost(self._cost, params)
            cst.append(cost)
        self.params = params
        return params, cst

    def predict(self, x):
        if self.params is None:
            raise ValueError('Model must be trained first!')
        return [self.qnode(features=features, parameters=self.params) for features in x]
