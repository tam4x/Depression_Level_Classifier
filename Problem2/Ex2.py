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

#copy= data_multiomics.groupby("Study Subject ID Number")

grouped = data_multiomics.groupby("Study Subject ID Number")

# Anzeigen der Gruppen
for name, group in grouped:
    print(f"Group: {name}")
    print(group)


#%% Model Training


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data_multiomics = data_multiomics[data_multiomics['timepoint'] != 4]






# Vorbereitung der Daten
X = data_multiomics["immune_system"]
y = data_multiomics['gestational_age']
groups = data_multiomics['Study Subject ID Number']

# Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()

# Modell definieren
model = SVC()

# Speicherung der Ergebnisse
accuracies = []



# LOOCV-Schleife
for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Modell trainieren
    model.fit(X_train, y_train)
    
    # Vorhersagen machen
    y_pred = model.predict(X_test)
    
    # Genauigkeit berechnen und speichern
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Durchschnittliche Genauigkeit
average_accuracy = np.mean(accuracies)
print(f"Durchschnittliche Genauigkeit: {average_accuracy:.2f}")

# Einzelne Genauigkeiten
print("Einzelne Genauigkeiten pro Durchlauf:")
print(accuracies)

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

