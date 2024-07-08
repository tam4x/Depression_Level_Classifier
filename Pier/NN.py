import sys
sys.path.append("..")
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
import itertools
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from NN import *
from sklearn.metrics import confusion_matrix

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

# Here the models are defined with different hidden layers and activation functions

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
    # TURN ON TRAINING MODE
    model.train()
    # TRAIN THE MODEL
    for epoch in range(num_epochs):
        running_loss = 0.0
        # enumerate over the data loader
        for i, (inputs, labels) in enumerate(train_dataloader):
            # perform the training loop -> forward, backward, optimize
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
def train_model_hyperparameter(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs, device):
    # Early stopping parameters
    patience = 10  # Number of epochs to wait if no improvement is observed
    min_delta = 0.001  # Minimum change in the monitored quantity to qualify as improvement
    best_val_loss = float('inf')
    current_patience = 0
    # train the model
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
    # evaluate the model and calculate the loss
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
        # calcualte accuracy and average loss
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(dataloader)
        print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy


#Function for hyperparameter tuning
def hyperparameter_tuning(train_dataloader, validation_dataloader, test_dataloader, criterion, num_epochs, device):
    # Hyperparameters being assigned here
    input_size = 56
    hidden_size = 128
    num_epochs = 30
    learning_rate_values = [0.001,0.003,0.006,0.01]
    momentum_values = [0.9, 0.95,0.99]
    results = {}
    optimizers = ['Adam', 'SGD']
    number = 1
    accuracys = []

    # Iterating over different models, optimizers and learning rates, momentum values to get the best model
    for i in range(3):
        for optimizer in optimizers:
            if optimizer == 'Adam':
                for learning_rate in learning_rate_values:
                    # creating the models
                    if i == 0:
                        m = Depression_Classifier_v_0(input_size, hidden_size).to(device)
                    elif i == 1:
                        m = Depression_Classifier_v_1(input_size, hidden_size).to(device)
                    else:
                        m = Depression_Classifier_v_2(input_size, hidden_size).to(device)
                    # choosing optimizer
                    o = optim.Adam(m.parameters(), lr=learning_rate)
                    # hyperparameter tuning
                    train_model_hyperparameter(m, train_dataloader,validation_dataloader,criterion, o, num_epochs, device)

                    validation_loss, accuracy = evaluate_model(m, test_dataloader, criterion, device)

                    # Store results
                    results[(f'Model_{i}', optimizer, learning_rate)] = {
                        'validation_loss': validation_loss,
                        'accuracy': accuracy
                    }
                    accuracys.append(accuracy)
                    print(f'Model: {number} is being trained with optimizer: {optimizer} and learning rate: {learning_rate} and Model_{i}')
                    number += 1
            else:
                for momentum, learning_rate in itertools.product(momentum_values, learning_rate_values):
                    if i == 0:
                        m = Depression_Classifier_v_0(input_size, hidden_size).to(device)
                    elif i == 1:
                        m = Depression_Classifier_v_1(input_size, hidden_size).to(device)
                    else:
                        m = Depression_Classifier_v_2(input_size, hidden_size).to(device)
                        
                    o = optim.SGD(m.parameters(), lr=learning_rate, momentum=momentum)
                    train_model_hyperparameter(m, train_dataloader,validation_dataloader,criterion, o, num_epochs, device)

                    validation_loss, accuracy = evaluate_model(m, test_dataloader, criterion, device)

                    # Store results
                    results[(f'Model_{i}', optimizer, learning_rate,momentum)] = {
                        'validation_loss': validation_loss,
                        'accuracy': accuracy
                    }
                    accuracys.append(accuracy)
                    print(f'Model: {number} is being trained with optimizer: {optimizer} and learning rate: {learning_rate} and Model_{i}')
                    number += 1
        
    return results, accuracys
    
# Function for evaluation
def evaluate(model, device, depression_feature):
    # Load the data
    if depression_feature == 'BP_PHQ_9':
            df = pd.read_csv('../data/Threshold_3_Operator_-_Depressionfeature_BP_PHQ_9_PercentofDataset_100.csv')
    
    elif depression_feature == 'MH_PHQ_S':
            df = pd.read_csv('../data/Threshold_15_Operator_-_Depressionfeature_MH_PHQ_S_PercentofDataset_100.csv')
     
    # Split the data into training and testing sets 0.8 is the training size and 0.2 is the testing size
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_features, test_targets = df_to_tensor(test_df, features_column='FEATURES', target_col='Depression')

    test_dataset = CustomDataset(test_features, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Test the model
    true_labels = []
    all_predicted = []
    predicted_probabilities = []
    for features, targets in test_dataloader:
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(features)
        predicted_probabilities.append(outputs.squeeze().tolist())
        #outputs = model_mh_phq_s(features)
        predicted = torch.round(outputs).int().squeeze().tolist()

        true_labels.append(targets.int().squeeze().tolist())
        all_predicted.append(predicted)

    true_labels_all = [item for sublist in true_labels for item in sublist]
    predicted_labels_all = [item for sublist in all_predicted for item in sublist]
    predicted_probabilities_all = [item for sublist in predicted_probabilities for item in sublist]
    # Generate confusion matrix
    cm = confusion_matrix(true_labels_all, predicted_labels_all)
    
    # Plot confusion matrix
    plt.figure(figsize=(9, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'Confusion_Matrix/Confusion_Matrix_MLP_{depression_feature}.png')
    TN, FP, FN, TP = cm.ravel()

    return true_labels_all, predicted_labels_all, predicted_probabilities_all, TN, FP, FN, TP



