import numpy as np
import pandas as pd
import random

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

def oversample(group, target_count):
    if len(group) < target_count:
        # Calculate how many samples we need
        samples_needed = target_count - len(group)
        # Randomly sample with replacement
        additional_samples = group.sample(samples_needed, replace=True)
        # Concatenate the original group with the additional samples
        return pd.concat([group, additional_samples])
    else:
        return group
    



    
