import sys
import os

# Add the path to the directory where your module is located
module_path = os.path.abspath(os.path.join('..', 'C:\\Users\\pier1\\OneDrive\\Desktop\\uni\\Master\\2.Semester\\Machine Learning (WIWI)\\Project\\Data for depression\\'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import your module
from helpers import *
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Convert DataFrame to PyTorch tensors
def df_to_tensor(df, features_column, target_col):
    columns = []
    features = []
    targets = []
    for indx, row in df.iterrows():
        feature = []
        for column in df.columns:
            if 'FEATURE' in column:
                feature.append(row[column])
        features.append(feature)
        targets.append(row[target_col])

 
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.int16)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.int16)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class Depression_Classifier_v_0(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Depression_Classifier_v_0, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out



class Depression_Classifier_v_1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Depression_Classifier_v_1, self).__init__()
        hidden_size_1 = int(hidden_size/2)
        hidden_size_2 = int(hidden_size/4)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size_1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

class Depression_Classifier_v_2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Depression_Classifier_v_2, self).__init__()
        hidden_size_1 = int(hidden_size/2)
        hidden_size_2 = int(hidden_size/4)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size_1)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden_size_2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Define the training function
def train_model(model, train_dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Define the training function
def train_model_cross_validation(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device):
    # Early stopping parameters
    patience = 10  # Number of epochs to wait if no improvement is observed
    min_delta = 0.001  # Minimum change in the monitored quantity to qualify as improvement
    best_val_loss = float('inf')
    current_patience = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                #print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 100:.4f}')
                #running_loss = 0.0
                pass

        # Calculate validation loss and monitor early stopping
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs,labels)  # Ensure labels are in correct shape
                val_loss += loss.item()
            val_loss /= len(validation_dataloader)
            
            # Print epoch statistics
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}')
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                    break

# Define the evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(dataloader)
        print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy

    





