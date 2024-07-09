# Depression Classifier

## Project Structure

- All problems are solved inside their respective folder
- The first problem is split between the EDA and the preprocessing-analysis in the .py files feature_extraction.py Data_Preprocessing_and_features.py helpers.py Data_processing.ipynb and the model training in the folders named after each member of our group. Inside these folders there are the machines, as well as folders with some (visual) results. 

    - Pier (Threshold 15 and 3)
            - FNN
            - Adaboost
            - Evaluation (Evaluation and creation of the Confusion Matrices and ROC Curve)
            - NN (NN Functions)
            - Confusion_Matrix (All the 4 Confusion Matrices -> 2 Datasets and 2 Models)
            - Models (For Neural Network)
            - Results (Of the Hyperparameter Tuning, was used to find the optimal Hyperparameters and the Metrices of the models)

        - Luisa (Threshold 17 and 2)
            - DT
            - RF
        - Ben (Threshold 19 and 6)
            - SVM
            - Elastic Net
        - Benedikt (Threshold 21 and 8)
            - Gradient_Boosting
            - XGBoost
        - Mo (Threshold 13 and 1)
            - Lasso
            - Ridge
            - Logistic Regression

    - Data_processing (Creation of the datasets)
    - helpers (Some helpers Functions)
    - Data_Preprocessing_and_features (File to create the processed Dataframe with parameters)
        - Threshold (For MH_PHQ_S = [13,15,17,19,21] and for BP_PHQ_9 = [1,2,3,6,8])
        - actigraphy_data_operator (+, -)
        - depression_classifier_feature (MH_PHQ_S, BP_PHQ_9)
        - percent_of_dataset (10,20,30,..,100) -> for slow PC ^^
    - EDA (for visual data exploration and analysis)
