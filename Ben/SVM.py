import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

# Initialize SVM classifier with RBF kernel (adjust parameters as needed)
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced' , probability=True)

# Train the SVM model
svm_classifier.fit(X_train, y_train)

# Make predictions and probabilities
y_pred_prob = svm_classifier.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
threshold = 0.5  # Example threshold

# Convert probabilities to binary predictions based on the threshold
y_pred = (y_pred_prob > threshold).astype(int)


# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate ROC AUC (requires predicted probabilities for positive class)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'ROC AUC: {roc_auc}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

#example results
#Accuracy: 0.9712787212787213
#Precision: 0.9154929577464789
#Recall: 0.3735632183908046
#F1-score: 0.5306122448979592
#ROC AUC: 0.9062693196482698