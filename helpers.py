import numpy as np
import pandas as pd
import random
import pywt
import numpy as np
from sklearn.preprocessing import StandardScaler
# Define functions to compute statistical features

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing target values
    
    for column in data.columns:
        if column == 'sex':
            # Count the number of NaN values
            nan_count = data[column].isna().sum()

            # Create a list of random values (1 or 2) with the same length as the number of NaN values
            random_values = [random.choice([1, 2]) for _ in range(nan_count)]
           
            # Create an iterator from the random values list
            random_values_iter = iter(random_values)

            # Replace NaN values with the random values
            data[column] = data[column].apply(lambda x: next(random_values_iter) if pd.isna(x) else x)

        elif column == 'year':
            data[column] = data[column].fillna(int(data[column].mean()))

        elif column in ['id','mod_d', 'ID']:
            pass

        else:
            data[column] = data[column].fillna(data[column].mean())

    return data


def Depression_Severity_(value): #for BP_PHQ_9
    if 0 <= value <= 4:
        return "None_disorder"
    elif value >= 5:
        return "Disorder"
    elif np.isnan(value):
        return "None_disorder"
    
def Depression_Severity(value):#for mh_PHQ_S
    if 0 <= value <= 4:
        return "None-minimal"
    elif 5 <= value < 10:
        return "Mild"
    elif 10 <= value < 15:
        return "Moderate"
    elif 15 <= value < 20:
        return "Moderately Severe"
    elif 20 <= value <= 27:
        return "Severe"
    elif np.isnan(value):
        return "None-minimal"
    
def BMI_range(value):
    if value < 18.5:
        return "underweight"
    elif 18.5 <= value < 25:
        return "Healthy Weight"
    elif 25 <= value < 30:
        return "overweight"
    elif value >= 30:
        return "obese"
    elif np.isnan(value):
        return "Healthy Weight"
    
def Sex_name(value):
    if value == 1:
        return "Male"
    elif value == 2:
        return "Female"
    elif np.isnan(value):
        return "Male"

def Age_range(value):
    if 19 <= value < 24:
        return "[19-23]"
    elif 24 <= value < 29:
        return "[24-28]"
    elif 29 <= value < 34:
        return "[29-33]"
    elif 34 <= value < 39:
        return "[34-38]"
    elif 39 <= value < 44:
        return "[39-43]"
    elif 44 <= value < 49:
        return "[44-48]"
    elif 49 <= value < 54:
        return "[49-53]"
    elif 54 <= value < 59:
        return "[54-58]"
    elif 59 <= value <= 65:
        return "[59-65]"
    elif np.isnan(value) == None:
        return "[34-38]"
    elif value > 65:
        return "[59-65]"
    elif value < 19:
        return "[19-23]"
    

def print_information(df):
    df_grouped = df.groupby('d_PHQ')
    for name, group in df_grouped:
        print(name)
        print(group.count())
    print('---Number of Non-Depression and Depression---')
    print(df.groupby('Depression').count()['ID_1'].iloc[0], df.groupby('Depression').count()['ID_1'].iloc[1])

def sampling(group, target_count, depression_feature):
    if depression_feature == 'BP_PHQ_9':
        if len(group) < target_count:
            # Calculate how many samples we need
            samples_needed = int((target_count - len(group)) / 4)
            # Randomly sample with replacement
            additional_samples = group.sample(samples_needed, replace=True)
            # Concatenate the original group with the additional samples
            balanced_group = pd.concat([group, additional_samples])
        elif len(group) > 10*target_count:
            # Undersampling: Randomly sample without replacement
            balanced_group = group.sample(10*target_count, replace=False)
        else:
            # Return the original group
            balanced_group = group

    elif depression_feature == 'MH_PHQ_S':
        if len(group) < target_count:
            # Calculate how many samples we need
            samples_needed = int((target_count - len(group)) / 2)
            # Randomly sample with replacement
            additional_samples = group.sample(samples_needed, replace=True)
            # Concatenate the original group with the additional samples
            balanced_group = pd.concat([group, additional_samples])
        elif len(group) > 4*target_count:
            # Undersampling: Randomly sample without replacement
            balanced_group = group.sample(4*target_count, replace=False)
        else:
            # Return the original group
            balanced_group = group

    return balanced_group

def sampling_v2(group, target_count):
    if len(group) < target_count:
        # Calculate how many samples we need
        samples_needed = target_count - len(group)
        # Randomly sample with replacement
        additional_samples = group.sample(samples_needed, replace=True)
        # Concatenate the original group with the additional samples
        balanced_group = pd.concat([group, additional_samples])
    elif len(group) >= target_count:
        # Undersampling: Randomly sample without replacement
        balanced_group = group.sample(target_count, replace=False)
  
    return balanced_group

def compute_energy(coeff):
    return np.sum(coeff ** 2)

def compute_mean(coeff):
    return np.mean(coeff)

def compute_std(coeff):
    return np.std(coeff)

def compute_entropy(coeff):
    p = np.abs(coeff) / np.sum(np.abs(coeff))
    return -np.sum(p * np.log2(p + np.finfo(float).eps))  # eps to avoid log(0)

def compute_features(data):
    coeffs = pywt.wavedec(data, 'db1')
    features = []
    for i, coeff in enumerate(coeffs):
        features.append(compute_energy(coeff))
        features.append(compute_mean(coeff))
        features.append(compute_std(coeff))
        features.append(compute_entropy(coeff))

    # Convert the feature list to a numpy array
    features = np.array(features)

    # (Optional) Normalize or standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    # print("Extracted features:")
    #print(features)

    return features


def compute_metrics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1
    



    
