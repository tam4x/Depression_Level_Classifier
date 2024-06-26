# Depression Classifier

## TODO
- Visualize Data to get correlations between features (Important)
- Oversample or Undersample Groups that are not well represented (maybe use undersampling, when threshold is really high. (BIAS))
- Use evaluation metrics (Confusion Matrix, ROC, F1 Score etc.)

## Explanation
- Survey 1 Datapoint per Participant
- New Patients have to be in the same group_pack and PAM[ID1] - PAM[ID2] -> new actigraphy data n^2 -n participants (75000 roughly)
- use the new actigraphy data to extract features from and add these features to the synthetic participant dataframe
- Use abs(PHQ9P2 - PHQ9P1) > threshold as a classification for depression or the abs(mh_PHQ_SP2 - mh_PHQ_SP1) > threshold as a classification


## Project Structure
- Folders
    - ALL (Here are the ALL-Files) -> needs to be created
    - HWP (Here is the weird data) -> needs to be created
    - PAM (Here is the actigraphy data or PAM data) -> needs to be created
    - data (here are the processed datasets located) -> needs to be created
    - Pier (Threshold (10 and 3))
        - Datasets
            - when using 10 as a threshold after sampling with method_1 72 % of the dataset are not depressed (38958 / (38958+14700))
            - when using 3 as a threshold after sampling with method_1 88 % of the dataset are not depressed (13346 / (13346 + 1802))
            - when using 10 as a threshold after sampling with method_1 35 % of the dataset are not depressed (13450 / (13450+24210))
            - when using 3 as a threshold after sampling with method_2 42 % of the dataset are not depressed (3228 / (3228 + 4304))
        - FNN
        - Adaboost         
    - Luisa (Threshold (12 and 2))
        - DT
        - RF
    - Ben (Threshold (14 and 6))
        - SVM
        - Elastic Net
    - Benedikt (Threshold (16 and 8))
        - Gradient_Boosting
        - XGBoost
    - Mo (Threshold (8 and 1))
        - Lasso
        - Ridge
        - Logistic Regression ?
- Data_processing (Only for Scripting functionality is in Synthetic_Patient)
- helpers (Some helpers Functions)
- Synthetic_Patient (File to create the processed Dataframe with parameters)
    - Threshold (For MH_PHQ_S = [8,10,12,14,16] and for BP_PHQ_9 = [1,2,3,6,8])
    - actigraphy_data_operator (+, -)
    - depression_classifier_feature (MH_PHQ_S, BP_PHQ_9)
    - percent_of_dataset (10,20,30,..,100) -> for slow PC ^^
