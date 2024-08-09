#!/bin/bash" -exec rm {} \;
find . -name "final_state_model.bin" -exec rm {} \;
find . -name "IQP_Full-Pauli-CRX_plot.svg" -exec rm {} \;
find . -name "IQP_Full-Pauli-CRX_results.json" -exec rm {} \;
find . -name "IQP_Full-Pauli-CRX_1D_plot.svg" -exec rm {} \;
find . -name "IQP_Full-Pauli-CRX.out" -exec rm {} \;
find . -name "IQP_Full-Pauli-CRX_predicted_values.csv" -exec rm {} \;
find . -name "model_log.csv" -exec rm {} \;
