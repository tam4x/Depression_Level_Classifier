import pandas as pd
import numpy as np
from helpers import *
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv('data/Threshold_4_Operator_+_Depressionfeature_MH_PHQ_S_PercentofDataset_50.csv')


print_information(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert DataFrame to PyTorch tensors
def df_to_tensor(df, features, target_col):
    features = df[features].values
    targets = df[target_col].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

train_features, train_targets = df_to_tensor(train_df,features='ACTIGRAPHY_DATA', target_col='Depression')
test_features, test_targets = df_to_tensor(test_df, features='ACTIGRAPHY_DATA', target_col='Depression')

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = CustomDataset(train_features, train_targets)
test_dataset = CustomDataset(test_features, test_targets)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)




