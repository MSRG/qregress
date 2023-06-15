import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
import joblib


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
            error_mitigation: list = None,
            shots=None):
        self.callback_interval = None
        self.x = None
        self.y = None
        self.params = None
        self.error_mitigation = error_mitigation
        self.num_qubits = num_qubits
        self._set_device(device, backend, shots)
        self.max_iterations = max_iterations
        self._set_optimizer(optimizer)
        self.pure = pure_qml
        self.encoder = encoder
        self.variational = variational
        self._build_qnode()
        self.qnode = qml.QNode(self._circuit, self.device)
        self.fit_count = 0

    def _set_device(self, device, backend, shots):
        #  sets the models quantum device. If using IBMQ asks for proper credentials
        if device == 'qiskit.ibmq':
            print('Running on IBMQ Runtime')
            instance = input('Enter runtime setting: instance')
            token = input('Enter IBMQ token')
            QiskitRuntimeService.save_account(channel='ibm_quantum', instance=instance, token=token, overwrite=True)
            self.device = qml.device(device, wires=self.num_qubits, backend=backend, shots=shots)
        else:
            self.device = qml.device(device, wires=self.num_qubits)

    def _set_optimizer(self, optimizer):
        #  sets the desired optimizer. SPSA is not available in scipy and has to be handled separately in fitting
        scipy_optimizers = ['COBYLA', 'Nelder-Mead']
        if optimizer in scipy_optimizers:
            self.optimizer = optimizer
            self.use_scipy = True
        else:
            self.optimizer = qml.SPSAOptimizer(maxiter=self.max_iterations)
            self.use_scipy = False

    def _circuit(self, features, parameters):
        #  builds the circuit with the given encoder and variational circuits.
        #  encoder and variational circuits must have only two required parameters, params/feats and wires
        self.encoder(features, wires=range(self.num_qubits))
        self.variational(parameters, wires=range(self.num_qubits))
        if self.pure:
            return qml.expval(qml.PauliZ(0))
        elif not self.pure:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _build_qnode(self):
        #  builds QNode from device and circuit using mitiq error mitigation if specified.
        #  TODO: Add more error mitigation options, specifically REM
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
        #  cost function for use in hybrid QML with linear model
        #  TODO: This isn't working all the time. Raising a matmul error.
        params = parameters[:-3] # change to num qubits
        extra_params = parameters[-3:]
        measurements = np.array([self.qnode(x, params) for x in self.x])
        cost = np.linalg.norm(self.y - np.matmul(measurements, extra_params))**2 / len(self.x)
        return cost

    def _num_params(self):
        #  computes the number of parameters required for the implemented variational circuit
        num_params = self.variational(None, wires=range(self.num_qubits), calc_params=True)
        return num_params

    def _save_partial_state(self, param_vector, force=False):
        # saves every fifth call to a bin file able to be loaded later by calling fit with load_state set to filename
        interval = self.callback_interval
        if interval is None:
            interval = 5
        if self.fit_count % interval == 0 or force:
            partial_results = param_vector
            outfile = 'partial_state.bin'
            joblib.dump(partial_results, outfile)
        self.fit_count += 1

    def _load_partial_state(self, infile):
        print('Loading partial state from file ' + infile)
        partial_state = joblib.load(infile)
        param_vector = partial_state
        print('Loaded parameter_vector as', param_vector)
        return param_vector

    def fit(self, x, y, initial_parameters=None, detailed_results=False, load_state=None, callback_interval=None):
        self.fit_count = 0
        self.callback_interval = callback_interval
        opt_result = None
        if load_state is not None:
            param_vector = self._load_partial_state(load_state)
            initial_parameters = param_vector
        elif initial_parameters is None:
            num_params = self._num_params()
            initial_parameters = np.random.rand(num_params)
        self.x = x
        self.y = y
        params = initial_parameters
        if self.pure:
            if self.use_scipy:
                opt_result = minimize(self._cost, x0=params, method=self.optimizer, callback=self._save_partial_state)
                self.params = opt_result['x']
            else:
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = self.optimizer.step_and_cost(self._cost, params)
                    cost.append(temp_cost)
                    self._save_partial_state(params)
                opt_result = [params, cost]
                self.params = params
        elif not self.pure:
            if self.use_scipy:
                opt_result = minimize(self._hybrid_cost, x0=params, method=self.optimizer,
                                      callback=self._save_partial_state)
            else:
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = self.optimizer.step_and_cost(self._hybrid_cost, params)
                    cost.append(temp_cost)
                    self._save_partial_state(params)
                opt_result = [params, cost]
        self._save_partial_state(params, force=True)
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
