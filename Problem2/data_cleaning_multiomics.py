# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:09:40 2024

@author: moham
"""



"""
Purpose: clean and cut the multiomics dataset to get just needed columns/features 
which do not contain any NaNs. The processed dataframe is exported as csv and is
used in this directory in the notebook "Problem 2".
"""
#%% Setup environment

#Import packages
import numpy as np
import pandas as pd
import pickle

# file path of dataset multiomics
file_path = r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\task1_multiomics_data.pickle"

# load pickle data into environment as a df
with open(file_path, "rb") as file:
    data_multiomics = pickle.load(file)

# show first 5 rows of df
print(data_multiomics.head(5))



#%% clean data by dropping columns

#delete multlevel index
data_multiomics.columns = data_multiomics.columns.droplevel(1)

# Delete columns, which are not needed for the first exercise of muliomics problem
data_multiomics = data_multiomics.drop("cellfree_rna", axis=1)
data_multiomics = data_multiomics.drop("metabolomics",  axis=1)
data_multiomics = data_multiomics.drop("microbiome", axis=1)
data_multiomics = data_multiomics.drop("plasma_luminex",  axis=1)
data_multiomics = data_multiomics.drop("serum_luminex", axis=1)
data_multiomics = data_multiomics.drop("plasma_somalogic",  axis=1)
data_multiomics = data_multiomics.drop("Training/Validation", axis=1)
data_multiomics = data_multiomics.drop("Gates ID", axis=1)
data_multiomics = data_multiomics.drop("MRN", axis=1)

# delete columns if they containing NaNs --> quick way, nans weren't present
data_multiomics = data_multiomics.dropna(axis=1, how='any')


#%% export processed data to files as csv
data_multiomics.to_csv(r"C:\Users\moham\OneDrive\Desktop\ML Uni Projekt\01 Code Project\Exercise 2\data_multinomics_processed.csv")
