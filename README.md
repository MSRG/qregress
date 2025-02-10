# qregress
Quantum kernels and regression

Ignore conflicts: `cat min_requirements.txt | xargs -n 1 pip install`

Use Viki's requirement.txt

TODO:


Example of how to run a function fitting for 5 qubits:

```
python3 main.py --settings function-calc-test/5qubits/A1-A1-CZ_Full-Pauli-CRX/A1-A1-CZ_Full-Pauli-CRX.json --train_set function-calc-test/linear/linear_train.bin --test_set function-calc-test/linear/linear_test.bin --scaler function-calc-test/linear/linear_scaler.bin >> check.out
```
