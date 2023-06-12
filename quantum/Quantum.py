import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory


class QuantumRegressor:

    def __init__(
            self,
            encoder,
            variational,
            num_qubits,
            optimizer='COBYLA',
            max_iterations=100,
            device='default.qubit',
            backend=None,
            pure_qml: bool = True,
            error_mitigation: list = None):
        if error_mitigation is None:
            self.error_mitigation = {
                'scale_factors': [1, 2, 3],
                'noise_scale_method': fold_global,
                'extrapolate': RichardsonFactory.extrapolate
            }
        self.num_qubits = num_qubits
        self._set_device(device, backend)
        self.max_iterations = max_iterations
        self._set_optimizer(optimizer)
        self.pure = pure_qml
        self.x = None
        self.y = None
        self.params = None
        self.encoder = encoder
        self.variational = variational
        self._build_qnode()
        self.qnode = qml.QNode(self._circuit, self.device)

    def _set_device(self, device, backend):
        if device == 'qiskit.ibmq':
            print('Running on IBMQ Runtime')
            instance = input('Enter runtime setting: instance')
            token = input('Enter IBMQ token')
            QiskitRuntimeService.save_account(channel='ibm_quantum', instance=instance, token=token, overwrite=True)
            self.device = qml.device(device, wires=self.num_qubits, backend=backend)
        else:
            self.device = qml.device(device, wires=self.num_qubits)
            self.error_mitigation = None

    def _set_optimizer(self, optimizer):
        scipy_optimizers = ['COBYLA', 'Nelder-Mead']
        if optimizer in scipy_optimizers:
            self.optimizer = optimizer
            self.use_scipy = True
        else:
            self.optimizer = qml.SPSAOptimizer(maxiter=self.max_iterations)
            self.use_scipy = False

    def _circuit(self, features, parameters):
        self.encoder(features, wires=range(self.num_qubits))
        self.variational(parameters, wires=range(self.num_qubits))
        if self.pure:
            return qml.expval(qml.PauliZ(0))
        elif not self.pure:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _build_qnode(self):
        self.qnode = qml.QNode(self._circuit, self.device)
        if self.error_mitigation is not None:
            scale_factors = self.error_mitigation['scale_factors']
            noise_scale_method = self.error_mitigation['noise_scale_method']
            extrapolate = self.error_mitigation['extrapolate']
            self.qnode = qml.transforms.mitigate_with_zne(self.qnode, scale_factors, noise_scale_method, extrapolate)

    def _cost(self, parameters):
        predicted_y = [self.qnode(x, parameters) for x in self.x]
        return mean_squared_error(self.y, predicted_y)

    def _hybrid_cost(self, parameters):
        params = parameters[:-3] # change to num qubits
        extra_params = parameters[-3:]
        measurements = np.array([self.qnode(x, params) for x in self.x])
        cost = np.linalg.norm(self.y - np.matmul(measurements, extra_params))**2 / len(self.x)
        return cost

    def _num_params(self):
        num_params = self.variational(None, wires=range(self.num_qubits), calc_params=True)
        return num_params

    def fit(self, x, y, initial_parameters=None, detailed_results=False):
        if initial_parameters is None:
            num_params = self._num_params()
            initial_parameters = np.random.rand(num_params)
        self.x = x
        self.y = y
        params = initial_parameters
        if self.pure:
            if self.use_scipy:
                opt_result = minimize(self._cost, x0=params, method=self.optimizer)
                self.params = opt_result['x']
            else:
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = self.optimizer.step_and_cost(self._cost, params)
                    cost.append(temp_cost)
                opt_result = (params, cost)
                self.params = params
        elif not self.pure:
            if self.use_scipy:
                opt_result = minimize(self._hybrid_cost, x0=params, method=self.optimizer)
            else:
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = self.optimizer.step_and_cost(self._hybrid_cost, params)
                    cost.append(temp_cost)
                opt_result = (params, cost)
        if detailed_results:
            return opt_result
        return self.params

    def predict(self, x):
        if self.params is None:
            raise ValueError('Model must be trained first!')
        if self.pure:
            return [self.qnode(features=features, parameters=self.params) for features in x]
        elif not self.pure:
            return [np.dot(self.qnode(features=features, parameters=self.params[:-3]), self.params[-3:]) for features in x]
