#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import click
import json
import time
import os
import itertools
import collections.abc
import pandas as pd
import pennylane as qml
from tqdm import tqdm
from qiskit_ibm_runtime import QiskitRuntimeService, Session

os.environ["OMP_NUM_THREADS"] = "12"

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
generator = np.random.default_rng(12958234)

# Synthetic data
X=generator.uniform(-1, 1, (100,5))
y=generator.uniform(-1, 1, (100,))
X_train, X_test, y_train, y_test = X[:80,:],X[80:,:],y[:80],y[80:]
print(X_train.shape,X_test.shape)



# Define custom Dataset
class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(targets).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
batch_size = 32 * 8
train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# Initial ize device and circuit
num_qubits=5
re_upload_depth=1
n_layers=1

service = QiskitRuntimeService(channel="ibm_quantum", instance='pinq-quebec-hub/univ-toronto/matterlab')
_backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
dev = qml.device("qiskit.remote", wires=num_qubits, backend=_backend,shots=1024,session=Session(backend=_backend))

# Qulacs simulator
# dev = qml.device("qulacs.simulator", wires=num_qubits)

# Simple circuit
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    for i in range(re_upload_depth):
        qml.AngleEmbedding(inputs, wires=range(num_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

# Initialize pytorch model
weight_shapes = {"weights": (n_layers, num_qubits)}
print(weight_shapes)
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(*[qlayer]).to(device)
print(model)


# Define loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(qlayer.parameters(), lr=learning_rate)

# Training loop
epochs = 10
train_losses = []
test_losses = []
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X=batch_X.to(device)
        batch_y=batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Evaluation on test set
    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_running_loss += loss.item() * batch_X.size(0)
    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_losses.append(test_epoch_loss)
    
    if (epoch+1) % 1 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Test Loss: {test_epoch_loss:.4f}")





# Evaluation
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(batch_y.squeeze().tolist())

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R$^2$: {r2:.4f}")

