import pennylane as qml
# from pennylane import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory, LinearFactory
import joblib
import mthree


class QuantumRegressor:
    """
    Machine learning model based on quantum circuit learning.

    Methods
    ------
    fit(x, y, initial_parameters=None, detailed_results=False, load_state=None, callback_interval=None)
        Fits the model instance to the given x and y data.
    predict(x)
        Predicts y values for a given array of input data based on previous training.

    """

    def __init__(
            self,
            encoder,
            variational,
            num_qubits,
            optimizer: str = 'COBYLA',
            max_iterations: int = 100,
            device: str = 'default.qubit',
            backend: str = None,
            postprocess: str = None,
            error_mitigation=None,
            scale_factors: list = None,
            folding=fold_global,
            shots: int = None,
            f: float = 1.,
            alpha: float = 0.):
        self.hyperparameters = {'f': f, 'alpha': alpha}
        if scale_factors is None:
            scale_factors = [1, 3, 5]
        self.callback_interval = None
        self.x = None
        self.y = None
        self.params = None
        self.error_mitigation = error_mitigation
        self.num_qubits = num_qubits
        self.max_iterations = max_iterations
        self.postprocess = postprocess
        self.encoder = encoder
        self.variational = variational
        self._set_device(device, backend, shots)
        self._set_optimizer(optimizer)
        self._build_qnode(scale_factors, folding)
        self.fit_count = 0

    def _set_device(self, device, backend, shots):
        #  sets the models quantum device. If using IBMQ asks for proper credentials
        if device == 'qiskit.ibmq':
            print('Running on IBMQ Runtime')
            # instance = input('Enter runtime setting: instance')
            # token = input('Enter IBMQ token')
            # QiskitRuntimeService.save_account(channel='ibm_quantum', instance=instance, token=token, overwrite=True)
            self.device = qml.device(device + '.circuit_runner', wires=self.num_qubits, backend=backend, shots=shots)
            service = QiskitRuntimeService()
            self._backend = service.backend(backend)
            if self.error_mitigation == 'TREX':
                self.device.set_transpile_args(**{'resilience_level': 1})
        else:
            self.device = qml.device(device, wires=self.num_qubits)

    def _set_optimizer(self, optimizer):
        #  sets the desired optimizer. SPSA is not available in scipy and has to be handled separately in fitting
        scipy_optimizers = ['COBYLA', 'Nelder-Mead']
        if optimizer in scipy_optimizers:
            self.optimizer = optimizer
            self.use_scipy = True
        elif optimizer == 'SPSA':
            self.use_scipy = False

    def _circuit(self, features, parameters):
        #  builds the circuit with the given encoder and variational circuits.
        #  encoder and variational circuits must have only two required parameters, params/feats and wires
        self.encoder(features, wires=range(self.num_qubits))
        self.variational(parameters, wires=range(self.num_qubits))
        if self.postprocess is None and self.error_mitigation != 'M3':
            return qml.expval(qml.PauliZ(0))
        elif self.postprocess is None and self.error_mitigation == 'M3':
            return [qml.counts(qml.PauliZ(0))]
        elif self.postprocess is not None and self.error_mitigation != 'M3':
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        elif self.postprocess is not None and self.error_mitigation == 'M3':
            return [qml.counts(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _build_qnode(self, scale_factors, folding):
        #  builds QNode from device and circuit using mitiq error mitigation if specified.
        self.qnode = qml.QNode(self._circuit, self.device)
        if self.error_mitigation == 'MITIQ_Linear':
            factory = LinearFactory.extrapolate
            scale_factors = scale_factors
            noise_scale_method = folding
            self.qnode = qml.transforms.mitigate_with_zne(self.qnode, scale_factors, noise_scale_method, factory)
        elif self.error_mitigation == 'MITIQ_Richardson':
            factory = RichardsonFactory.extrapolate
            scale_factors = scale_factors
            noise_scale_method = folding
            self.qnode = qml.transforms.mitigate_with_zne(self.qnode, scale_factors, noise_scale_method, factory)
        elif self.error_mitigation == 'M3':
            mit = mthree.M3Mitigation(self._backend)
            mit.cals_from_system()
            old_qnode = self.qnode

            def new_qnode(features, params):
                raw_counts = old_qnode(features, params)
                m3_counts = [mit.apply_correction(raw_counts[i], [i], return_mitigation_overhead=False)
                             for i in range(len(raw_counts))]
                expval = [counts.expval() for counts in m3_counts]
                if len(expval) == 1:
                    expval = expval[0]
                return expval

            self.qnode = new_qnode

    def _cost(self, parameters):
        #  f is a hyperparameter scaling each of the obtained measurements used in both pure and hybrid
        f = self.hyperparameters['f']
        predicted_y = f * np.array([self.qnode(x, parameters) for x in self.x])
        return mean_squared_error(self.y, predicted_y)

    def _hybrid_cost(self, parameters):
        #  cost function for use in hybrid QML with linear model
        #  TODO: This isn't working all the time. Raising a matmul error.
        f = self.hyperparameters['f']
        alpha = self.hyperparameters['alpha']
        num = self.num_qubits
        params = parameters[:-num]
        extra_params = parameters[-num:]
        #  f is a hyperparameter scaling each of the obtained measurements used in both pure and hybrid
        measurements = f * np.array([self.qnode(x, params) for x in self.x])
        base_cost = np.linalg.norm(self.y - np.matmul(measurements, extra_params)) ** 2 / len(self.x)
        if self.postprocess == 'simple':
            cost = base_cost
        elif self.postprocess == 'ridge':
            ridge_lambda = alpha
            cost = base_cost + ridge_lambda * np.linalg.norm(extra_params)
        elif self.postprocess == 'lasso':
            lasso_lambda = alpha
            num = 0
            for param in extra_params:
                num += np.abs(param)
            cost = base_cost + lasso_lambda * num
        elif self.postprocess == 'elastic':
            num = 0
            elastic_lambda = 1
            for param in extra_params:
                num += np.abs(param)
            cost = base_cost + elastic_lambda * (alpha * num + (1 - alpha) * np.linalg.norm(extra_params))
        else:
            raise ValueError('Unable to determine classical postprocessing method.' +
                             'postprocess was set to ', self.postprocess, " accepted values include: " +
                             " 'simple', 'ridge', 'lasso', 'elastic'")
        return cost

    def _num_params(self):
        #  computes the number of parameters required for the implemented variational circuit
        num_params = self.variational.num_params
        return num_params

    def _save_partial_state(self, param_vector, force=False):
        # saves every fifth call to a bin file able to be loaded later by calling fit with load_state set to filename
        interval = self.callback_interval
        if interval is None:
            interval = 5
        if self.fit_count % interval == 0 or force:
            partial_results = param_vector
            if force is True:
                outfile = 'final_state' + '.' + self.optimizer + '.' \
                          + '.' + self.encoder.__name__
            else:
                outfile = 'partial_state' + self.optimizer + self.encoder.__name__
            joblib.dump(partial_results, outfile)
        self.fit_count += 1

    def _load_partial_state(self, infile):
        print('Loading partial state from file ' + infile)
        partial_state = joblib.load(infile)
        param_vector = partial_state
        print('Loaded parameter_vector as', param_vector)
        return param_vector

    def fit(self, x, y, initial_parameters=None, detailed_results=False, load_state=None, callback_interval=None):
        """
        Fits the current model to the given x and y data. If no initial parameters are given then random ones will be
        chosen. Optimal parameters are stored in the model for use in predict and returned in this function.

        :param x: np.array
            x data to fit
        :param y: np.array
            y data to fit
        :param initial_parameters: list, optional
            initial parameters to start optimizer
        :param detailed_results: bool, optional
            whether to return detailed results of optimization or just parameters
        :param load_state: str, optional
            file to load partial fit data from
        :param callback_interval: int, optional
            how often to save the optimization steps to file
        :return:
            returns the optimal parameters found by optimizer. If detailed_results=True and optimizer is scipy, then
            will be of type scipy optimizer results stored in dictionary.
        """
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
        if self.postprocess is None:
            if self.use_scipy:
                opt_result = minimize(self._cost, x0=params, method=self.optimizer, callback=self._save_partial_state)
                self.params = opt_result['x']
            else:
                opt = qml.SPSAOptimizer(maxiter=self.max_iterations)
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = opt.step_and_cost(self._cost, params)
                    cost.append(temp_cost)
                    self._save_partial_state(params)
                opt_result = [params, cost]
                self.params = params
        elif self.postprocess is not None:
            if self.use_scipy:
                opt_result = minimize(self._hybrid_cost, x0=params, method=self.optimizer,
                                      callback=self._save_partial_state)
            else:
                opt = qml.SPSAOptimizer(maxiter=self.max_iterations)
                cost = []
                for _ in range(self.max_iterations):
                    params, temp_cost = opt.step_and_cost(self._hybrid_cost, params)
                    cost.append(temp_cost)
                    self._save_partial_state(params)
                opt_result = [params, cost]
        self._save_partial_state(params, force=True)
        if detailed_results:
            return opt_result
        return self.params

    def predict(self, x):
        """
        Predicts a set of output data given a set of input data x using the trained parameters found with fit

        :param x: np.array
            x data to predict outputs of in the model
        :raises ValueError:
            if fit is not first called then raises error explaining that the model must first be trained
        :return: list
            predicted values corresponding to each datapoint in x
        """
        if self.params is None:
            raise ValueError('Model must be trained first!')
        if self.postprocess is None:
            return [self.qnode(features=features, parameters=self.params) for features in x]
        elif self.postprocess is not None:
            return [np.dot(self.qnode(features=features, parameters=self.params[:-self.num_qubits]),
                           self.params[-self.num_qubits:]) for features in x]
