# Depression Classifier

## TODO
-Add means or generate values for the EDA Functions. Use np.mean for Series or use an list of possible values like [Male, Female, Male, ...] and turn it into and iterator (can be improved)
- Visualize Data to get correlations between features
- Group Data by Sex -> Age_Range -> bmi_range -> count
- Create synthetic participants base on the created groups
- create binary variable by using abs(PHQ_9P2 - PHQ_9P1) > treshhold
- what are the possible features of the input vector?
- search for possible neural networks that can be finetuned
- use pretrained- CNN or Neural Networks in general to extract features from actigraphy data?
- combine these features with the features given in the ALL Dataframe
