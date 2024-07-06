# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:09:40 2024

@author: moham
"""

import numpy as np
import pandas as pd
import pickle

# Geben Sie den vollständigen Pfad zur Datei an
file_path = r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\task1_multiomics_data.pickle"

# Laden der Pickle-Datei
with open(file_path, "rb") as file:
    data_multiomics = pickle.load(file)

# Anzeigen der ersten 5 Zeilen des DataFrames
print(data_multiomics.head(5))



#%%
data_multiomics.to_csv(r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\data_multinomics.csv")

#%%

#data_multiomics.to_excel(r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\data_multinomics.xlsx")

'''
Evaluationsmetriken:   
-Spearman-Korrelation
-Mean Absolute Error (MAE) 
-Root Mean Squared Error (RMSE)
'''
data_multiomics.columns = data_multiomics.columns.droplevel(1)



#%%
data_multiomics = data_multiomics.drop("cellfree_rna", axis=1)
data_multiomics = data_multiomics.drop("metabolomics",  axis=1)
data_multiomics = data_multiomics.drop("microbiome", axis=1)
data_multiomics = data_multiomics.drop("plasma_luminex",  axis=1)
data_multiomics = data_multiomics.drop("serum_luminex", axis=1)
data_multiomics = data_multiomics.drop("plasma_somalogic",  axis=1)

#%%




#data_multiomics = data_multiomics.applymap(lambda x: ''.join(filter(lambda y: y in set(printable), str(x))) if isinstance(x, str) else x)

#data_multiomics = pd.DataFrame(data_multiomics)
#data_multiomics.loc[data_multiomics['timepoint'] == 4, 'gestational_age'] = 0


#%%
import string

data_multiomics.dropna(axis=1)

data_multiomics.columns = data_multiomics.columns.str.strip()


#print(data_multiomics.dtypes)

printable = set(string.printable)

def clean_data(value):
    if isinstance(value, str):
        return ''.join(filter(lambda x: x in printable, value))
    return value

data_multiomics = data_multiomics.applymap(clean_data)



#data_multiomics.loc[data_multiomics['timepoint'] == 4, 'gestational_age'] = 0
#%%
data_multiomics['timepoint'].astype(int)
data_multiomics['gestational_age'].astype(int)
data_multiomics['MRN'].astype(int)
data_multiomics['Study Subject ID Number'].astype(int)
#data_multiomics['sex bin'].astype(int)

def convert_to_float_and_fill_na(df):
    for col in df.columns[7:]:  # Ab der achten Spalte (Index 7)
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            df[col] = np.nan
    return df

# Anwenden der Funktion auf den DataFrame
data_multiomics = convert_to_float_and_fill_na(data_multiomics)

data_multiomics.dropna(axis=1)

#%%

data_multiomics = data_multiomics.drop("Training/Validation", axis=1)
data_multiomics = data_multiomics.drop("Gates ID", axis=1)
data_multiomics = data_multiomics.drop("MRN", axis=1)

#%%
data_multiomics = data_multiomics.drop("Sex", axis=1)

#%%



# Werte der Spalte 'gestational_age' auf 0 setzen bei den angegebenen Indizes
#copy = data_multiomics[['timepoint', 'gestational_age']]


#copy.loc[copy.index % 4 == 3, 'gestational_age'] = 0


copy= data_multiomics.loc[data_multiomics.index % 4 == 3, 'gestational_age'] = 0

#%%
data_multiomics.to_csv(r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\data_multinomics_processed.csv")

#%%

copy = data_multiomics

# required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extractor(df, threshold=0.9):
    """
    ************************************************************************
    EXPLANATIONS

    Filter features based on correlation threshold and visualize the correlation matrices.

    In practice a threshold of 0.9 is widespread but there exists more conservatve
    thresholds like 0.8 which is a more conservative approach. You may choose it
    depending on the context.

    ************************************************************************

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the features.
    threshold (float): The correlation threshold to filter features. Default is 0.9.

    Returns:
    selected_features (list): List of features with correlation below the threshold.
    excluded_features (list): List of features with correlation above the threshold.
    """

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Create lists for selected features ("normal correlations") and excluded features (perfectly/almost perfectly correlated)
    selected_features = []
    excluded_features = []

    # Run through the upper triangular matrix of the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                excluded_features.append(colname)

    # fill lists of excluded and selected features
    excluded_features = list(set(excluded_features))
    selected_features = [feature for feature in corr_matrix.columns if feature not in excluded_features]

    # create correlation matrix of selected features
    selected_corr_matrix = corr_matrix.loc[selected_features, selected_features]

    # plot correlation matrix of selected features
    plt.figure(figsize=(12, 10))
    sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Selected Features')
    plt.show()

    # create correlation matrix of excluded features and plot it
    if excluded_features:
        remaining_corr_matrix = corr_matrix.loc[excluded_features, excluded_features]
        plt.figure(figsize=(20, 16))
        sns.heatmap(remaining_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Excluded Features')
        plt.show()
    else:
        print("No excluded features found.")

    print("Selected Features:", selected_features)
    print("Excluded Features:", excluded_features)


    # return lists of selected and excluded features
    return selected_features, excluded_features

extractor(data_multiomics)


#%%
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


X = data_multiomics["immune_system"]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Modell definieren
model = SVC()

# Speicherung der Ergebnisse
accuracies = []
maes = []
rmses = []
spearman_corrs = []

# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    print(f"Train-Index: {train_index}, Test-Index: {test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Modell trainieren
    model.fit(X_train, y_train)
    
    # Vorhersagen machen
    y_pred = model.predict(X_test)
    
    # Genauigkeit berechnen und speichern
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # MAE berechnen und speichern
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    
    # RMSE berechnen und speichern
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    
    # Spearman-Korrelation berechnen und speichern
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_corrs.append(spearman_corr)

# Durchschnittliche Genauigkeit
average_accuracy = np.mean(accuracies)
print(f"Durchschnittliche Genauigkeit: {average_accuracy:.2f}")

# Durchschnittliche MAE
average_mae = np.mean(maes)
print(f"Durchschnittliche MAE: {average_mae:.2f}")

# Durchschnittliche RMSE
average_rmse = np.mean(rmses)
print(f"Durchschnittliche RMSE: {average_rmse:.2f}")

# Durchschnittliche Spearman-Korrelation
average_spearman = np.mean(spearman_corrs)
print(f"Durchschnittliche Spearman-Korrelation: {average_spearman:.2f}")

# Einzelne Genauigkeiten
print("Einzelne Genauigkeiten pro Durchlauf:")
print(accuracies)

# Einzelne MAEs
print("Einzelne MAEs pro Durchlauf:")
print(maes)

# Einzelne RMSEs
print("Einzelne RMSEs pro Durchlauf:")
print(rmses)

# Einzelne Spearman-Korrelationen
print("Einzelne Spearman-Korrelationen pro Durchlauf:")
print(spearman_corrs)

#%%% First try with elastic net

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


X = data_multiomics.loc[:, data_multiomics.columns.str.startswith("immune_system")]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Modell definieren
model = ElasticNet()

# Speicherung der Ergebnisse
maes = []
rmses = []
spearman_corrs = []

# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    print(f"Train-Index: {train_index}, Test-Index: {test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Modell trainieren
    model.fit(X_train, y_train)
    
    # Vorhersagen machen
    y_pred = model.predict(X_test)
    
    # MAE berechnen und speichern
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    
    # RMSE berechnen und speichern
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    
    # Spearman-Korrelation berechnen und speichern
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_corrs.append(spearman_corr)

# Durchschnittliche MAE
average_mae = np.mean(maes)
print(f"Durchschnittliche MAE: {average_mae:.2f}")

# Durchschnittliche RMSE
average_rmse = np.mean(rmses)
print(f"Durchschnittliche RMSE: {average_rmse:.2f}")

# Durchschnittliche Spearman-Korrelation
average_spearman = np.mean(spearman_corrs)
print(f"Durchschnittliche Spearman-Korrelation: {average_spearman:.2f}")

# Einzelne MAEs
print("Einzelne MAEs pro Durchlauf:")
print(maes)

# Einzelne RMSEs
print("Einzelne RMSEs pro Durchlauf:")
print(rmses)

# Einzelne Spearman-Korrelationen
print("Einzelne Spearman-Korrelationen pro Durchlauf:")
print(spearman_corrs)


#%% 2nd try with elastic net and hyperparameter tuning

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr



X = data_multiomics.loc[:, data_multiomics.columns.str.startswith("immune_system")]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Hyperparameter-Raster für ElasticNet
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

# Funktion zur Durchführung des Hyperparameter-Tunings
def hyperparameter_tuning(X_train, y_train):
    model = ElasticNet()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Speicherung der Ergebnisse
maes = []
rmses = []
spearman_corrs = []

# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    print(f"Train-Index: {train_index}, Test-Index: {test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Hyperparameter-Tuning durchführen
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Vorhersagen machen
    y_pred = best_model.predict(X_test)
    
    # MAE berechnen und speichern
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    
    # RMSE berechnen und speichern
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    
    # Spearman-Korrelation berechnen und speichern
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_corrs.append(spearman_corr)

# Durchschnittliche MAE
average_mae = np.mean(maes)
print(f"Durchschnittliche MAE: {average_mae:.2f}")

# Durchschnittliche RMSE
average_rmse = np.mean(rmses)
print(f"Durchschnittliche RMSE: {average_rmse:.2f}")

# Durchschnittliche Spearman-Korrelation
average_spearman = np.mean(spearman_corrs)
print(f"Durchschnittliche Spearman-Korrelation: {average_spearman:.2f}")

# Einzelne MAEs
print("Einzelne MAEs pro Durchlauf:")
print(maes)

# Einzelne RMSEs
print("Einzelne RMSEs pro Durchlauf:")
print(rmses)

# Einzelne Spearman-Korrelationen
print("Einzelne Spearman-Korrelationen pro Durchlauf:")
print(spearman_corrs)



#%% elastic net with LOOCV and hyperparameter tuning


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Assuming `data_multiomics` is your DataFrame
# Example: data_multiomics = pd.read_csv('your_file.csv')

# Extracting the relevant columns
X = data_multiomics.loc[:, data_multiomics.columns.str.startswith("immune_system")]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Hyperparameter grid for ElasticNet
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

# Speicherung der Ergebnisse
maes = []
rmses = []
spearman_corrs = []

# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    print(f"Train-Index: {train_index}, Test-Index: {test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Erstellen eines GridSearchCV mit LeaveOneGroupOut
    grid_search = GridSearchCV(estimator=ElasticNet(), param_grid=param_grid, scoring='neg_mean_absolute_error', cv=logo.split(X_train, y_train, groups[train_index]))
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Vorhersagen machen
    y_pred = best_model.predict(X_test)
    
    # MAE berechnen und speichern
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    
    # RMSE berechnen und speichern
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    
    # Spearman-Korrelation berechnen und speichern
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_corrs.append(spearman_corr)

# Durchschnittliche MAE
average_mae = np.mean(maes)
print(f"Durchschnittliche MAE: {average_mae:.2f}")

# Durchschnittliche RMSE
average_rmse = np.mean(rmses)
print(f"Durchschnittliche RMSE: {average_rmse:.2f}")

# Durchschnittliche Spearman-Korrelation
average_spearman = np.mean(spearman_corrs)
print(f"Durchschnittliche Spearman-Korrelation: {average_spearman:.2f}")

# Einzelne MAEs
print("Einzelne MAEs pro Durchlauf:")
print(maes)

# Einzelne RMSEs
print("Einzelne RMSEs pro Durchlauf:")
print(rmses)

# Einzelne Spearman-Korrelationen
print("Einzelne Spearman-Korrelationen pro Durchlauf:")
print(spearman_corrs)


"""
Results:
    0: 
    1
    2
    3
"""

#%% PCA for dimension reduction and random forest and hyperparameter tuning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

# Assuming `data_multiomics` is your DataFrame
# Example: data_multiomics = pd.read_csv('your_file.csv')

# Extracting the relevant columns
X = data_multiomics.loc[:, data_multiomics.columns.str.startswith("immune_system")]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Hyperparameter grid for PCA and RandomForest
param_grid = {
    'pca__n_components': [0.9, 0.95, 0.99],  # Keep 90%, 95%, 99% of variance
    'randomforest__n_estimators': [100, 200],
    'randomforest__max_features': ['auto', 'sqrt', 'log2']
}

# Pipeline for PCA and RandomForest
pipeline = Pipeline(steps=[
    ('pca', PCA()),
    ('randomforest', RandomForestRegressor())
])

# Speicherung der Ergebnisse
maes = []
rmses = []
spearman_corrs = []

# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    print(f"Train-Index: {train_index}, Test-Index: {test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Erstellen eines GridSearchCV mit LeaveOneGroupOut
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=logo.split(X_train, y_train, groups[train_index]))
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Vorhersagen machen
    y_pred = best_model.predict(X_test)
    
    # MAE berechnen und speichern
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    
    # RMSE berechnen und speichern
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    
    # Spearman-Korrelation berechnen und speichern
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_corrs.append(spearman_corr)

# Durchschnittliche MAE
average_mae = np.mean(maes)
print(f"Durchschnittliche MAE: {average_mae:.2f}")

# Durchschnittliche RMSE
average_rmse = np.mean(rmses)
print(f"Durchschnittliche RMSE: {average_rmse:.2f}")

# Durchschnittliche Spearman-Korrelation
average_spearman = np.mean(spearman_corrs)
print(f"Durchschnittliche Spearman-Korrelation: {average_spearman:.2f}")

# Einzelne MAEs
print("Einzelne MAEs pro Durchlauf:")
print(maes)

# Einzelne RMSEs
print("Einzelne RMSEs pro Durchlauf:")
print(rmses)

# Einzelne Spearman-Korrelationen
print("Einzelne Spearman-Korrelationen pro Durchlauf:")
print(spearman_corrs)

