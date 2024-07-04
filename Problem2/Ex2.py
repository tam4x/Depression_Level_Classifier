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
data_multiomics.loc[data_multiomics['timepoint'] == 4, 'gestational_age'] = 0















