import time
import json
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
from qiskit_ibm_runtime import (Batch, EstimatorV2 as Estimator)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit_ibm_runtime")
warnings.filterwarnings("ignore",category=DeprecationWarning,module="qiskit_aer.backends.aer_compiler")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def map2qiskit(quantum_circuit, num_qubits, X):
    '''
    Map features to Qiskit Circuit
    
    parameters
    ----------
    quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Qiskit Quantum Circuit
        
    num_qubits: int
        Number of qubits
        
    X: numpy.ndarray
        Feature matrix

    returns
    -------
    quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Qiskit Quantum Circuit
    
    '''
    featparameters = dict([(i,X[idx % num_qubits]) for idx,i in enumerate(quantum_circuit.parameters) if 'x' in i.name])
    quantum_circuit = quantum_circuit.assign_parameters(featparameters)
    return quantum_circuit


def get_results(job):
    '''
    Get the results of a quantum job from IBM Quantum
    
    parameters
    ----------
    job: qiskit.primitives.primitive_job.PrimitiveJob
        Job submitted to the IBM Quantum device

    returns
    -------
    pred: numpy.ndarray
        Predicted values
    
    '''
    result = job.result()
    pred = np.vstack([r.data.evs for r in result]).flatten()
    return pred    

def batched_pred(parameters, quantum_circuit, hamiltonian, num_qubits, X, backend, shots, resilience_level, n_jobs):
    '''
    Function to predict quantum circuits in batches

    parameters
    ----------
    parameters: numpy.ndarray
        Initial circuit parameters
        
    quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Qiskit Quantum Circuit

    hamiltonian: list
        List containing the observables
        
    num_qubits: int
        Number of qubits
        
    X: numpy.ndarray
        Feature matrix
        
    backend: qiskit_ibm_runtime backend
        Backend of choice (real or fake)
        
    shots: float or int
        Number of shots. Default is 1024.0.
    
    resilience_level: int
        Error mitigation level. Default is 1.

    n_jobs: int
        Number of threads/cores for parallelization.

    returns
    -------
    y_pred: numpy.ndarray
        Predicted values    
    '''
    mapped_circuits = [[map2qiskit(quantum_circuit,num_qubits,x_i) for x_i in x] for x in X]
    tiled_parameters = np.tile(parameters,(X.shape[0],X.shape[1])).reshape(X.shape[0],X.shape[1],-1)
    
    # Submit jobs
    jobs = []
    t0 = time.perf_counter()
    for idx, (anz, pars) in enumerate(zip(mapped_circuits,tiled_parameters)):
        with Batch(backend=backend) as batch:
            if 'fake' in backend.backend_name:
                jobid = str(idx)
            else:
                jobid = batch.details()['id']
            estimator = Estimator(mode=batch)
            estimator.options.default_shots = shots
            estimator.options.resilience_level = resilience_level
            pub = [(a, [hamiltonian], [p]) for a, p in zip(anz, pars)]
            result = estimator.run(pubs=pub)
            jobs.append((jobid,result))
    print(f"Submitted to device in {time.perf_counter()-t0:.4f} s")
    
    t1 = time.perf_counter()
    y_pred = np.vstack(joblib.Parallel(n_jobs=1,verbose=0, prefer="threads")(joblib.delayed(get_results)(job) for jobid, job in tqdm(jobs,desc="Running batch: ")))
    print(f"Predicted in {time.perf_counter()-t1:.4f} s")

    return y_pred    


def cost_func(parameters, quantum_circuit, hamiltonian, num_qubits, X, y, cost_history_dict, backend, shots=1024.0, resilience_level=1, n_jobs=1):
    """
    Cost function. Is this loss?
    
    parameters
    ----------
    parameters: numpy.ndarray
        Initial circuit parameters
        
    quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Qiskit Quantum Circuit

    hamiltonian: list
        List containing the observables
        
    num_qubits: int
        Number of qubits
        
    X: numpy.ndarray
        Feature matrix

    y: numpy.ndarray
        Target vector

    cost_history_dict: dict
        Dictionary to track the loss        
        
    backend: qiskit_ibm_runtime backend
        Backend of choice (real or fake)
        
    shots: float or int
        Number of shots. Default is 1024.0.
    
    resilience_level: int
        Error mitigation level. Default is 1.

    n_jobs: int
        Number of threads/cores for parallelization.

    """
    t0=time.perf_counter()
    
    y_pred = batched_pred(parameters, quantum_circuit, hamiltonian, num_qubits, X, backend, shots, resilience_level, n_jobs)
    
    loss = mean_squared_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = parameters
    cost_history_dict["cost_history"].append(loss)
    
    print(f"Iters. done: {cost_history_dict['iters']} Current cost: {loss} Accuracy: {r2} Time: {time.perf_counter()-t0}")
    
    with open('model_log.csv', 'a') as outfile:
        log = f"{time.asctime()},{cost_history_dict['iters']},{loss},{parameters}\n"
        outfile.write(log)

    save_file = 'partial_state_model.bin'
    progress = {'x': parameters, 'loss': loss}
    joblib.dump(progress, save_file)
  
    return loss    

def evaluate(parameters, quantum_circuit, hamiltonian, num_qubits, n_jobs, backend, X_train, y_train, X_test=None, y_test=None, plot: bool = False, title: str = 'defult',y_scaler=None, shots=1024.0, resilience_level=1):
    scores = {}
    st = time.time()
    print('Now scoring model... ')
    
    y_train_pred = batched_pred(parameters, quantum_circuit, hamiltonian, num_qubits, X_train, backend, shots, resilience_level, n_jobs)
    y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1))

    scores['MSE_train'] = mean_squared_error(y_train, y_train_pred)
    scores['R2_train'] = r2_score(y_train, y_train_pred)
    scores['MAE_train'] = mean_absolute_error(y_train, y_train_pred)

    y_test_pred = None
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    if y_test is not None:
        y_test_pred = batched_pred(parameters, quantum_circuit, hamiltonian, num_qubits, X_test, backend, shots, resilience_level, n_jobs)
        y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
        scores['MSE_test'] = mean_squared_error(y_test, y_test_pred)
        scores['R2_test'] = r2_score(y_test, y_test_pred)
        scores['MAE_test'] = mean_absolute_error(y_test, y_test_pred)

    if plot:
        plt.figure()
        if y_test_pred is not None:
            plt.scatter(y_test, y_test_pred, color='b', s=10, label=f'Test, MAE = {scores["MAE_test"]:.2f}')
        plt.scatter(y_train, y_train_pred, color='r', s=10, label=f'Train, MAE = {scores["MAE_train"]:.2f}')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.axis('scaled')

        max_val = max(max(plt.xlim()), max(plt.ylim()))
        plt.xlim((0, max_val))
        plt.ylim((0, max_val))

        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        plt.plot([x_min, x_max], [y_min, y_max], 'k--', alpha=0.2, label='y=x')
        plt.legend()
        plt.savefig(title+'_plot.svg')

        if X_test.shape[1] == 1:
            plt.figure()
            plt.title(title)
            plt.scatter(X_train, y_train_pred, color='b', label='Train', s=10)
            plt.scatter(X_test, y_test_pred, color='orange', label='Test', s=10)
            plt.scatter(X_train, y_train, color='green', label='Data', s=10)
            plt.scatter(X_test, y_test, color='green', s=10)
            plt.legend()
            plt.savefig(title+'_1D_plot.svg')

    print(f'Scoring complete taking {time.time() - st} seconds. ')

    return scores, y_test_pred, y_train_pred    


def run(parameters, quantum_circuit, hamiltonian, num_qubits, X_train, y_train, X_test, y_test, scaler, backend, iters=1, shots=1024.0, resilience_level=1, n_jobs=1):
    """
    Cost function. Is this loss?
    
    parameters
    ----------
    parameters: numpy.ndarray
        Initial circuit parameters
        
    quantum_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        Qiskit Quantum Circuit

    hamiltonian: list
        List containing the observables
        
    num_qubits: int
        Number of qubits
        
    X_train: numpy.ndarray
        Feature matrix

    y_train: numpy.ndarray
        Target vector
        
    X_test: numpy.ndarray
        Feature matrix

    y_test: numpy.ndarray
        Target vector

    scaler: sklearn.preprocessing._data.MinMaxScaler
        Scikit-learn scaler
        
    backend: qiskit_ibm_runtime backend
        Backend of choice (real or fake)

    iters: int
        Number of optimization iterations
        
    shots: float or int
        Number of shots. Default is 1024.0.
    
    resilience_level: int
        Error mitigation level. Default is 1.

    n_jobs: int
        Number of threads/cores for parallelization.

    """
    scores = []
    with open('model_log.csv', 'w') as outfile:
        outfile.write('Time,Iteration,Cost,Parameters')
        outfile.write('\n')        
    
    
    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }
    
    
    res = minimize(cost_func,
        parameters,
        args=(quantum_circuit, hamiltonian, num_qubits, X_train, y_train, cost_history_dict, backend, shots, resilience_level, n_jobs),
        method="cobyla", options={'maxiter':iters})  

    parameters = res.x
    loss = res.fun    
    progress = {'x': parameters, 'loss': loss}
    joblib.dump(progress, 'final_state_model.bin')
    os.remove('partial_state_model.bin') 
    
    scores, y_test_pred, y_train_pred = evaluate(parameters, quantum_circuit, hamiltonian, num_qubits, n_jobs, backend, X_train, y_train, X_test=X_test, y_test=y_test, plot = True, title= 'A2_HWE-CNOT',y_scaler=scaler, shots=shots, resilience_level=resilience_level)
    
    name = 'A2_HWE-CNOT_predicted_values.csv'
    train_pred, y_train, test_pred, y_test = y_train_pred.flatten().tolist(), y_train.flatten().tolist(), y_test_pred.flatten().tolist(), y_test.flatten().tolist()
    df_train = pd.DataFrame({'Predicted': train_pred, 'Reference': y_train})
    df_train['Data'] = 'Train'
    df_test = pd.DataFrame({'Predicted': test_pred, 'Reference': y_test})
    df_test['Data'] = 'Test'
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[['Data', 'Predicted', 'Reference']]
    
    df.to_csv(name, index=False)

    results_title = 'A2_HWE-CNOT_results.json'
    with open(results_title, 'w') as outfile:
        json.dump(scores, outfile)


    