#!/bin/bash
find . -name "*.out" -exec rm {} \;
find . -name "*.svg" -exec rm {} \;
find . -name "*_predicted_values.csv" -exec rm {} \;
find . -name "*_results.json" -exec rm {} \;
find . -name "final_state_model.bin" -exec rm {} \;
find . -name "model_log.csv" -exec rm {} \;
