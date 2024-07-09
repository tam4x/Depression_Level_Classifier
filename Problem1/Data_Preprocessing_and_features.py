import pandas as pd
import numpy as np
from helpers import *
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import operator
import pywt

operator_map = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '**': operator.pow,
}

class Synthetic_Patient_Dataset:

    def __init__(self, threshold : int, actigraphy_data_operator: str, depression_classifier_feature: str, percent_of_dataset: int):
        # Goes from -27 to 27 so with absolute from 0 - 27 possible thresholds = [12, 15, 18, 20, 22, 24]
        self.threshold = threshold
        # Operators include + - * / as a string
        self.operator = actigraphy_data_operator
        # classifier Feature has to be string and BP_PHQ_9 or MH_PHQ_S BP_PHQ_1 -> BP_PHQ_8 can also be added 
        self.depression_feature = depression_classifier_feature

        self.percent = percent_of_dataset

    def load_data(self, path_all, path_pam):
        print(f'Loading Datasets from {path_all} and {path_pam}')
        self.all14_df=pd.read_sas(path_all + 'hn14_all.sas7bdat')
        self.all16_df=pd.read_sas(path_all + 'hn16_all.sas7bdat')
        self.pam14_df=pd.read_sas(path_pam + 'HN14_PAM.sas7bdat')
        self.pam16_df=pd.read_sas(path_pam + 'hn16_pam.sas7bdat')
        
    def remove_features(self):
        print('Removing Features')
        self.all16_df = self.all16_df[["ID", "year", "sex", "age", "BP_PHQ_9",
                  "mh_PHQ_S", "HE_BMI", "mh_stress", "EQ5D"]]
        self.all14_df = self.all14_df[["id", "year", "sex", "age", "BP_PHQ_9",
                        "mh_PHQ_S", "HE_BMI", "mh_stress", "EQ5D"]]
        
        self.all14_df, self.all16_df = process_data(self.all14_df), process_data(self.all16_df)
    
    def create_intervalls(self):
        print('Creating Intervalls')
        self.all14_df['HE_BMI'], self.all16_df['HE_BMI'] = self.all14_df['HE_BMI'].apply(BMI_range), self.all16_df['HE_BMI'].apply(BMI_range)
        self.pam14_df['sex'], self.pam16_df['sex'], self.all14_df['sex'], self.all16_df['sex'] = self.pam14_df['sex'].apply(Sex_name), self.pam16_df['sex'].apply(Sex_name), 				  self.all14_df['sex'].apply(Sex_name), self.all16_df['sex'].apply(Sex_name)
        self.pam14_df['age'], self.pam16_df['age'], self.all14_df['age'], self.all16_df['age'] = self.pam14_df['age'].apply(Age_range), self.pam16_df['age'].apply(Age_range), self.all14_df['age'].apply(Age_range), self.all16_df['age'].apply(Age_range)

    def process_data(self):
        print('Processing Data')
        # Create a function to rename columns to uppercase
        func = lambda df: df.rename(columns=str.upper)
        # Apply the function to each dataframe
        self.pam14_df, self.pam16_df, self.all14_df, self.all16_df = map(func, [self.pam14_df, self.pam16_df, self.all14_df, self.all16_df])
        # Concentrate the dataframes and remove the MOD_D column
        self.pam_combined = pd.concat([self.pam14_df, self.pam16_df], ignore_index=True)
        self.all_combined = pd.concat([self.all14_df, self.all16_df], ignore_index=True)
        self.pam_combined.drop('MOD_D', axis=1, inplace=True)
        # Transform the ID column to string from bytestring
        self.pam_combined['ID'] = self.pam_combined['ID'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.all_combined['ID'] = self.all_combined['ID'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    def create_Synthetic_Dataset(self):
        print('Creating Synthetic Patients')
        # Group the PAM data by ID
        self.pam_grouped = self.pam_combined.groupby('ID')
        # Cut the dataset to the desired percentage
        cut = int(len(self.all_combined) * (self.percent / 100))
        self.all_combined = self.all_combined.iloc[1:cut]
        
        # Create an empty list to store pairs of IDs
        id_pairs = []
        group_names = []
        sex_names = []
        age_names = []
        bmi_names = []
        PHQ_value = np.array([])
        # Iterate over each group
        for name, group in self.all_combined.groupby(['SEX', 'AGE', 'HE_BMI']):
            # Get IDs in the group
            ids = group['ID'].tolist()
            valid_ids = []

            for id1 in ids:
                # Get the valid IDs that are included in the PAM data
                try:
                    data_participant_1 = self.pam_grouped.get_group(id1)['PAXINTEN'].to_numpy() # this is for checking if it is a valid ID
                    valid_ids.append(id1)
                except KeyError:
                    pass
            # iterate over the valid_ids to create pairs
            for id_1 in valid_ids:
                for id_2 in valid_ids:
                    if id_1 == id_2: #or (id_2,id_1) in id_pairs:
                        pass
                    else:
                        # Get the IDs and add them to the list aswell as the group names
                        id_pairs.append((id_1,id_2))
                        group_names.append(name[0] + '_' + name[1] + '_' + name[2])
                        sex_names.append(name[0])
                        age_names.append(name[1])
                        bmi_names.append(name[2])

                        # PHQ9P1 = all_combined.loc[all_combined['ID'] == id_1, 'BP_PHQ_9'].iloc[0]
                        # PHQ9P2 = all_combined.loc[all_combined['ID'] == id_2, 'BP_PHQ_9'].iloc[0]
                        # Get the PHQ9 values for the IDs
                        PHQSP1 = self.all_combined.loc[self.all_combined['ID'] == id_1, self.depression_feature].iloc[0]
                        PHQSP2 = self.all_combined.loc[self.all_combined['ID'] == id_2, self.depression_feature].iloc[0]
                        # Create the values based on the absolute difference
                        value = abs(int(PHQSP1 - PHQSP2))
                        PHQ_value = np.append(PHQ_value, value)

        # Create a dataframe from the created lists
        self.id_pairs_df = pd.DataFrame(id_pairs, columns=['ID_1', 'ID_2'])
        self.id_pairs_df['group_id'] = group_names
        self.id_pairs_df['SEX'] = sex_names
        self.id_pairs_df['AGE'] = age_names
        self.id_pairs_df['HE_BMI'] = bmi_names
        self.id_pairs_df['ID_COMBINED'] = self.id_pairs_df['ID_1'] + self.id_pairs_df['ID_2']
        self.id_pairs_df['d_PHQ'] = PHQ_value
        # Here the values are transformed into binary values because the threshold is set to a specific value
        self.id_pairs_df['Depression'] = (self.id_pairs_df['d_PHQ'] >= self.threshold).astype(int)

    def calculate_actigraphy(self):

        print('Calculating Actigraphy Data from Synthetic Patients')
        pam_synthetic = pd.DataFrame(columns=['ID','ACTIGRAPHY_DATA'], dtype = object)
        # Create an empty array with zeros to store the synthetic data
        synthetic_array = np.zeros((self.id_pairs_df.shape[0], 10080)) # 10080 number of samples for a single patient
        id_combined = []
        number = 0

        for index,synthetic_patient in self.id_pairs_df.iterrows():
            # Get the data for the participants 1 and 2
            data_participant_1 = self.pam_grouped.get_group(synthetic_patient['ID_1'])['PAXINTEN'].to_numpy()
            data_participant_2 = self.pam_grouped.get_group(synthetic_patient['ID_2'])['PAXINTEN'].to_numpy()
            # Apply the operator to the data
            op_func = operator_map[self.operator]
            result = op_func(data_participant_1, data_participant_2)
            # Store the absolute value of the result divided by 2
            synthetic_array[number] = np.abs(result/2)

            id_combined.append(synthetic_patient['ID_1'] + synthetic_patient['ID_2'])
            logging.info(f"Participant_1 {synthetic_patient['ID_1']} and Participant_2 {synthetic_patient['ID_2']} added with {synthetic_array[number]}")
            number += 1
        # Create the Combined ID and ACTIGRAPHY_DATA columns
        pam_synthetic['ID'] = id_combined
        mask = []
        # Delete the rows where the actigraphy data is zero at all datapoints
        for row in range(synthetic_array.shape[0]):
            max_value = np.max(synthetic_array[row, :])
            if max_value == 0 or max_value == 0.0:
                mask.append(row)
        synthetic_array = np.delete(synthetic_array, mask, axis=0)
        # Add the actigraphy data to the dataframe based on the rows
        for row in range(synthetic_array.shape[0]):
            pam_synthetic.at[row, 'ACTIGRAPHY_DATA'] = synthetic_array[row]
        self.id_pairs_df['ACTIGRAPHY_DATA'] = pam_synthetic['ACTIGRAPHY_DATA']

    def plot_data(self, index):
        plt.plot(self.id_pairs_df['ACTIGRAPHY_DATA'].iloc[index])

    def save_data(self, path):
        print(f'Saving Data into {path}')
        self.id_pairs_df.to_csv(path, index=False)

    def compute_features(self):
        feature_list = []
        # Iterate over the participants and calculate the features with wavelet transformation
        for index,participant in self.id_pairs_df.iterrows():
            features = compute_features(participant['ACTIGRAPHY_DATA'])
            feature_list.append(features)

        feature_list = np.array(feature_list, dtype=np.float32)
        # Add the features to the dataframe
        for i in range(feature_list.shape[1]):
            self.id_pairs_df[f'FEATURE_{i}'] = feature_list[:, i]
       
    def remove_actigraphy(self):
        self.id_pairs_df.drop('ACTIGRAPHY_DATA', axis=1, inplace=True)

    def particicipant_distribution(self, before_sampling = True, sampler: int = 1):
        grouped = self.id_pairs_df.groupby('d_PHQ')
        grouped_counts = []
        grouped_names = []
        for name, group in grouped:
            grouped_counts.append(group.shape[0])
            grouped_names.append(name)
        # Plotting the counts
        plt.figure(figsize=(10, 6))
        plt.bar(grouped_names, grouped_counts, color='skyblue')

        # Adding titles and labels
        plt.title('Counts per Group for d_phq')
        plt.xlabel('Participant Group')
        plt.ylabel('Count')

        if before_sampling:
            plt.savefig(f'data/Participant_Distribution_before_sampling_{self.depression_feature}_{sampler}.png')
        else:
            plt.savefig(f'data/Participant_Distribution_after_sampling_{self.depression_feature}_{sampler}.png')
        # Show the plot
        plt.show()

    def dataset_sample(self):
        # Apply the oversampling function to each group
        if self.depression_feature == 'BP_PHQ_9':
            self.id_pairs_df = self.id_pairs_df.groupby('d_PHQ').apply(lambda x: sampling_small(x, 80, self.depression_feature)).reset_index(drop=True)
        else:
            self.id_pairs_df = self.id_pairs_df.groupby('d_PHQ').apply(lambda x: sampling_small(x, 40, self.depression_feature)).reset_index(drop=True)

