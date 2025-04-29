import numpy as np
import time
import joblib
from joblib import dump, load
import os
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from qiskit_ibm_runtime import Batch

class QiskitRegressor:
    def __init__(self,
                 quantumcircuit,
                 numqubits,
                 ansatz_layers,
                 reuploaddepth,
                 backend,
                 observables_labels,
                 channel = None,
                 instance = None,
                 parameterpath = None,
                 optimization_level = 2,
                 resilience_level = 1,
                 shots = 1024,
                 iterations = 1000,
                 verbose = False,
                 n_jobs = None
                ):
        '''
        parameters
        ----------
        quantumcircuit: 
            Qiskit quantum circuit

        numqubits: int
            Number of qubits before mapping
            
        ansatz_layers: int
            Number of ansatz layers (AL)

        reuploaddepth: int
            Re-upload depth (RUD)

        backend: str
            Options: statevector, fake, real

        observables_labels: str
            Observables mapped to original circuit before ISA pass

        channel: str
            IBM Quantum channel

        instance: str
            IBM Quantum instance

        parameterpath: str
        
        optimization_level: int
            Qiskit circuit optimization level (default=2; details at "https://docs.quantum.ibm.com/guides/set-optimization")
            
        resilience_level: int
            Qiskit error mitigation level (default=2; details at "https://docs.quantum.ibm.com/guides/configure-error-mitigation"       
        
        shots: int
            Number of shots (multiples of 1024 preferred)

        iterations: int
            Number of training cycles

        verbose: bool
            Printing level. True prints all, False prints nothing

        n_jobs: int
            Number of cores/threads
            
        '''
        self.qc = quantumcircuit
        self.AL = ansatz_layers
        self.RUD = reuploaddepth
        self.num_qubits = numqubits
        self.backendstr = backend
        self.observables_labels = observables_labels
        self.channel = channel
        self.instance = instance
        self.parameterpath = parameterpath
        self.optimization_level = optimization_level
        self.resilience_level = resilience_level
        self.shots = shots 
        self.iterations = iterations
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        self._initial_parameters()
        self._setdevice()
        
        
    def _initial_parameters(self):
        '''
        Load or create model parameters
        '''
        print(self.parameterpath)
        if self.parameterpath==None:
            if self.verbose:
                print('Parameters from scratch')
            num_params = len([i for i in list(self.qc.parameters) if 'theta' in i.name]) // self.AL
            generator = np.random.default_rng(12958234)            
            self.x0 = np.tile(generator.uniform(-np.pi, np.pi, num_params),self.AL*self.RUD) 
        else:
            if self.verbose:
                print('Parameters loaded')
            try:
                self.x0 = np.load(self.parameterpath)['x0']
            except:
                self.x0 =np.array(joblib.load(self.parameterpath)['x'])
        print(self.x0)

    def _setdevice(self):
        '''
        Sets the device
        '''
        if self.backendstr=='statevector':
            observables = [SparsePauliOp(self.observables_labels)]
            self.qc = self.qc
            self.mapped_observables  = [observable.apply_layout(self.qc.layout) for observable in observables]
            self.service = None
            self._backend = self.backendstr
                
        else:
            # Select backend, you need to grab the whole device
            self.service = QiskitRuntimeService(channel=self.channel, instance=self.instance)
            self._backend = self.service.least_busy(operational=True, simulator=False, min_num_qubits=127)
            self.maxcircuits = self._backend.max_circuits
            self.target = self._backend.target
            if self.backendstr=='fake':
                # generate a simulator that mimics the real quantum
                # system with the latest calibration results
                self._backend = AerSimulator.from_backend(self._backend)            
                # self._backend = FakeQuebec()
                self.target = self._backend.target
                
            # Generate pass manager
            pm = generate_preset_pass_manager(target=self.target, optimization_level=self.optimization_level)
            self.qc = pm.run(self.qc)
            
            observables = [SparsePauliOp(self.observables_labels)]
            
            self.mapped_observables = [observable.apply_layout(self.qc.layout) for observable in observables]
               
            
        if self.verbose:
            print(self.mapped_observables)
            print(self._backend)

    def _map2qiskit(self, X):
        '''
        Map features to Qiskit Circuit
        
        parameters
        ----------        
        X: numpy.ndarray
            Feature matrix
    
        returns
        -------
        ansatz: qiskit.circuit.quantumcircuit.QuantumCircuit
            Qiskit Quantum Circuit
        
        '''
        qc = self.qc
        featparams = dict([(i,X[idx % self.num_qubits]) for idx,i in enumerate(qc.parameters) if 'x' in i.name])

        qc = qc.assign_parameters(featparams)
        return qc

    def get_results(self,filename=None,job=None):
        '''
        Get the results of a quantum job from IBM Quantum
        
        parameters
        ----------
        filename: str/None
            Name of file

        job: list
            List of list containing jobs
    
        returns
        -------
        pred: numpy.ndarray
            Predicted values
        
        '''
        if self.backendstr=='statevector' or self.backendstr=='fake':
            pred = np.hstack([np.vstack([r.data.evs for r in rs.result()]) for jid, rs in job])
        else:
            with open(filename,'r') as f:
                pred = np.hstack([np.vstack([r.data.evs for r in self.service.job(jid.strip()).result()]) for jid in f.readlines()])
             
        return pred   

    def _batchmap(self,index,ansatz,params,file=None):
        '''
        Function to map batches
    
        parameters
        ----------
        index: int
            Index of batch
            
        ansatz: qiskit.circuit.quantumcircuit.QuantumCircuit
            Qiskit Quantum Circuit
        
        params: numpy.ndarray
            Model parameters
            
        file: class
            _io.TextIOWrapper
    
        returns
        -------
        y_pred: numpy.ndarray
            Predicted values    
        '''    
        results = []
        
        if isinstance(self._backend, str):
            jobid = str(index)
            estimator = StatevectorEstimator()
            for a1, p1 in zip(ansatz, params):
                pub = [(a, [self.mapped_observables], [p]) for a, p in zip(a1,p1)]
                result = estimator.run(pubs=pub)
                results.append([jobid,result])
                           
        else:
            with Batch(backend=self._backend,max_time='8h') as batch:
                if self._backend.name!='ibm_quebec':
                    jobid = str(index)
                else:
                    jobid = batch.details()['id']
                estimator = Estimator(mode=batch)
                estimator.options.default_shots = self.shots
                estimator.options.resilience_level = self.resilience_level
            
                        
                for a1, p1 in zip(ansatz, params):
                    pub = [(a, [self.mapped_observables], [p]) for a, p in zip(a1,p1)]
                    result = estimator.run(pubs=pub)
                    results.append([result.job_id(),result])
                    file.write(f"{result.job_id()}\n")
    
        return results    

    def predict(self,X,parameters=None,iters=None,restart=False,filename=None):
        '''
        Function to predict quantum circuits in batches
    
        parameters
        ---------- 

        X: numpy.ndarray
            Feature matrix

        parameters: numpy.ndarray
            Model parameters
                            
        iters: int/str
            Helps label the jobs.txt file
    
        restart: bool
            Read *txt file unless Default=False
    
        returns
        -------
        y_pred: numpy.ndarray
            Predicted values    
        '''
        if parameters is None:
            parameters = self.x0 

        if restart==False:
            mapped_circuits = [[[self._map2qiskit(x_i) for x_i in xi ] for xi in X[i : i + 4]] for i in range(0, len(X), 4)]
            tiled_params = np.tile(parameters,(X.shape[0],X.shape[1])).reshape(X.shape[0],X.shape[1],-1)
            batched_params = [tiled_params[i : i + 4] for i in range(0, len(X), 4)]
            #  Set some parameters to assist with writing to file
            if isinstance(self._backend, str):
                filename = None
                file = None
            else:
                filename = f'jobs_{iters}.txt'
                file = open(filename,'w')            
            # Submit jobs
            t0 = time.perf_counter()
            jobs = joblib.Parallel(n_jobs=self.n_jobs,verbose=0, prefer="threads")(joblib.delayed(self._batchmap)(idx,anz,pars,file) for idx, (anz, pars) in tqdm(enumerate(zip(mapped_circuits,batched_params)),desc="Mappings"))
            
            if isinstance(self._backend, str)==False:
                # Close file after writing            
                file.close() 
            
            if self.verbose:
                print(f"Submitted to device in {time.perf_counter()-t0:.4f} s")

        

            self.jobs = jobs
                      
            t1 = time.perf_counter()
            if self.backendstr=='statevector' or self.backendstr=='fake':
                y_pred = np.hstack(joblib.Parallel(n_jobs=self.n_jobs,verbose=0, prefer="threads")(joblib.delayed(self.get_results)(filename,job) for job in tqdm(self.jobs,desc="Running batch: "))).T
            else:
                y_pred = self.get_results(filename)
            
            if self.verbose:
                print(f"Predicted in {time.perf_counter()-t1:.4f} s")          
                
        else:
            if filename is None:
                raise TypeError
            else:
                t1 = time.perf_counter()        
                y_pred = np.hstack(self.get_results(filename)).T
                if self.verbose:
                    print(f"Predicted in {time.perf_counter()-t1:.4f} s") 
                    
        print("return",y_pred.shape)
        return y_pred      

    def _cost_func(self,parameters,X, y):
        """
        Cost function. Is this loss?
        
        parameters
        ----------            
        parameters: numpy.ndarray
            Model parameters
            
        X: numpy.ndarray
            Feature matrix
    
        y: numpy.ndarray
            Target vector
    
        cost_history_dict: dict
            Dictionary to track the loss                    
    
        restart: bool
            Read *txt file unless Default=False        
    
        """
        t0=time.perf_counter()
        
        y_pred = self.predict(X,parameters,iters=self.cost_history_dict["iters"])
        y_pred = np.nan_to_num(y_pred) 
        loss = mean_squared_error(y.flatten(),y_pred.flatten())
        r2 = r2_score(y.flatten(),y_pred.flatten())
        
        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["prev_vector"] = parameters
        self.cost_history_dict["cost_history"].append(loss)
        
        if self.verbose:
            print(f"Iters. done: {self.cost_history_dict['iters']} Current cost: {loss} Accuracy: {r2} Time: {time.perf_counter()-t0}")
        
        with open('model_log.csv', 'a') as outfile:
            log = f"{time.asctime()},{self.cost_history_dict['iters']},{loss},{parameters}\n"
            outfile.write(log)
    
        save_file = 'partial_state_model.bin'
        progress = {'x': parameters, 'loss': loss}
        dump(progress, save_file)
        self.x0=parameters
        self.loss=loss

        return loss    

    def fit(self,X,y):
        '''
        Fit the regressor and dump data
        
        parameters
        ----------
        X: numpy.ndarray
            Features
        y: numpy.ndarray
            Target values
        '''
        
        scores = []
        with open('model_log.csv', 'w') as outfile:
            outfile.write('Time,Iteration,Cost,Parameters')
            outfile.write('\n')        
        
        
        self.cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
            
        
        # for iters in range(self.iterations):
        res = minimize(self._cost_func,
            self.x0,
            args=(X, y),
            method="cobyla", options={'maxiter':self.iterations})  

   
            
        progress = {'x': self.x0, 'loss': self.loss}
        dump(progress, 'final_state_model.bin')
        os.remove('partial_state_model.bin') 
        
