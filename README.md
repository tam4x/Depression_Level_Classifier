# Depression Classifier

## Project Structure
Problem 1: Depression LevelClassification Using Machine Learning

The first problem is split between the preprocessing and analysis in the .py files feature_extraction.py Synthetic_Patient.py helpers.py Data_processing.ipynb and the model training in the folders named after each member of our group. Inside these folders there are the machines, as well as folders with some (visual) results, that we'll further compare and analyse in our presentation. 

- Pier (Threshold (15 and 3))
        - Datasets
            - when using 10 as a threshold after sampling with method_1 72 % of the dataset are not depressed (38958 / (38958+14700))
            - when using 3 as a threshold after sampling with method_1 88 % of the dataset are not depressed (13346 / (13346 + 1802))
            - when using 10 as a threshold after sampling with method_1 35 % of the dataset are not depressed (13450 / (13450+24210))
            - when using 3 as a threshold after sampling with method_2 42 % of the dataset are not depressed (3228 / (3228 + 4304))
        - FNN
        - Adaboost
        - Evaluation (Evaluation and creation of the Confusion Matrices and ROC Curve)
        - NN (NN Functions)
        - Confusion_Matrix (All the 8 Confusion Matrices -> 4 Datasets and 2 Models)
        - Models (For Neural Network)
        - Results (Of the Hyperparameter Tuning, was used to find the optimal Hyperparameters and the Metrices of the models)       
    - Luisa (Threshold (17 and 2))
        - DT
        - RF
    - Ben (Threshold (19 and 6))
        - SVM
        - Elastic Net
    - Benedikt (Threshold (21 and 8))
        - Gradient_Boosting
        - XGBoost
    - Mo (Threshold (13 and 1))
        - Lasso
        - Ridge
        - Logistic Regression ?    
- Data_processing (Only for Scripting functionality is in Synthetic_Patient)
- helpers (Some helpers Functions)
- Synthetic_Patient (File to create the processed Dataframe with parameters)
    - Threshold (For MH_PHQ_S = [13,15,17,19,21] and for BP_PHQ_9 = [1,2,3,6,8])
    - actigraphy_data_operator (+, -)
    - depression_classifier_feature (MH_PHQ_S, BP_PHQ_9)
    - percent_of_dataset (10,20,30,..,100) -> for slow PC ^^

The other problems are solved inside their respective folder. 
