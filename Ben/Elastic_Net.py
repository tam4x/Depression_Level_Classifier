import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('/home/jovyan/Depression_Level_Classifier/data/Threshold_6_Operator_-_Depressionfeature_BP_PHQ_9_PercentofDataset_100.csv')

columns_with_features = [col for col in data.columns if "FEATURE" in col]

# Define features (X) and target (y)
X = data[columns_with_features]
y = data['Depression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1896)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model with hyperparameters
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=1896)

# Train the model
elastic_net.fit(X_train, y_train)

# Make predictions
y_pred = elastic_net.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# example results 

#Mean Squared Error: 0.04588164249964719
#R^2 Score: -7.100615565236446e-05

#Mean Squared Error: 0.0430201479714974
#R^2 Score: 0.06230029434645268