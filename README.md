# Depression Classifier

## TODO
- Visualize Data to get correlations between features
- Oversample Groups that are not well represented
- Create a Neural Network
- what are the possible features of the input vector?
- search for possible neural networks that can be finetuned
- use pretrained- CNN or Neural Networks in general to extract features from actigraphy data?
- combine these features with the features given in the ALL Dataframe

## Explanation
- Survey 1 Datapoint per Participant
- New Patients have to be in the same group_pack and PAM[ID1] - PAM[ID2] -> new actigraphy data n^2 -n participants (75000 roughly)
- use the new actigraphy data to extract features from and add these features to the synthetic participant dataframe
- Use abs(PHQ9P2 - PHQ9P1) > threshhold as a classification for depression or the (mh_PHQ_SP2 + mh_PHQ_SP1) / 2 as a classification


## Project Structure
- Folders
    - ALL
    - HWP
    - PAM
    - data
- Data_processing (Only for Scripting functionality is in Synthetic_Patient)
- helpers (Some helpers Functions)
- NN (File for Neural Network)
- Synthetic_Patient (File to create the processed Dataframe with parameters)
    - Threshold (Better to use one under 15 because then there are to many Non-Depression Patients in Dataset)
    - actigraphy_data_operator (+, -, *, /)
    - depression_classifier_feature (MH_PHQ_S, BP_PHQ_9 etc.)
    - percent_of_dataset (10,20,30,..,100)