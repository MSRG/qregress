sed -i 's/"OPTIMIZER": "COBYLA"/"OPTIMIZER": "SPSA"/g' */*json
sed -i 's/"MAX_ITER": 2000/"MAX_ITER": 1000/g' */*json
