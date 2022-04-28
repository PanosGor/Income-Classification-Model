# Income-Classification-Model
A Semi-Supervised classification model predicting if an individual's annual income is more or less than $50K

## Goal

The goal of this project is to create a ML model that predicts if a person’s salary will be above or below 50K dollars

## Preprocessing Phase

Three different datasets were provided for the classification part of this project. 
- test-project.data
- train-project.data
- unlabeled-project.data

The preprocessing phase includes all the steps in order to clear the datasets from duplicate values null values and prepare it for the model training and testing phase.

*Starting with the train-project.data first part was to check the dataset*


![image](https://user-images.githubusercontent.com/82097084/165744531-da39791d-d6a2-4a2d-aba7-8c413b6ccdc1.png)

Figure above shows a general information with regards to the dataset. As can be seen the dataset consisted of 10000 rows and 15 different columns. 
The dataset consisted of both numerical and nominal values.

*Numerical*
-	Age
-	Final_Weight
-	Education_Num
-	Capital_Gain
-	Capital_Loss
-	Hours_per_week

*Nominal*
-	WorkClass
-	Education
-	Marital_Status
-	Occupation
-	Relationship
-	Race
-	Sex
-	Native_Country
-	Class

**Data Cleaning**

First a test was made by using Seaborn library in order to make a visual representation of all the null values in the dataset.


![image](https://user-images.githubusercontent.com/82097084/165746030-c7f295a0-089c-4cff-94e9-c9422f2b99c4.png)

Figure above represents all the null values in the whole dataset before preprocessing. 
In total 755 found to have null values in the whole dataset (only 7.55% of the total data).

![image](https://user-images.githubusercontent.com/82097084/165746231-ec0b32d6-6042-4f8f-b9d5-c2235e609835.png)

After removing them all null values, 9245 values remained in the dataset. 

**Handling Categorical Values**

The analysis shown the number of unique labels for each categorical column as can be seen below.
-	WorkClass – 7 unique labels
-	Education – 16 unique labels
-	Marital_Status – 7 unique labels
-	Occupation – 14 unique labels
-	Relationship – 6 unique labels
-	Race – 5 unique labels
-	Sex – 2 unique labels
-	Native_Country – 40 unique labels
-	Class – 2 unique labels

Pandas built in function get_dummies was used in order to make the transformation for the above columns. 
The result was to transform the dataset successfully, but the dataset’s dimensions were increased significantly as well. 
More specifically the dimensions before using get_dummies were (9245,15) while after using get_dummies the dimensions changed to (9245,105). 
90 extra columns were added in the dataset.

*Native_Country*

*Frequency per Country*

![image](https://user-images.githubusercontent.com/82097084/165746809-17e8ffd3-b82a-48a7-be58-f47e6025623c.png)

The figure above shows a barplot of the frequency of the data for each country
The first distribution that was checked was that of the native country as can be seen most of the values in the dataset belong to the United States. 
As a result the rest of the countries could be grouped altogether under a single category leaving only 2 choices either the Native_country would “United States ” or not.

The same exactly methodology was followed for the rest of the categorical values as well. 

*Frequency per Education Category*

![image](https://user-images.githubusercontent.com/82097084/165746973-ebc0ea32-c3a1-4896-a7d6-190180dadef0.png)

As can be seen from the figure above the first four categories Highschool graduate, Bachelors, Some Certificate have the most values compared to the rest categories all the rest of the categories were grouped under a new category other. 
Phd remained as a separate category even though not many observations were falling under, the reason was the importance of that a Phd title would have to a person’s salary

*Frequency per Occupation Category*

![image](https://user-images.githubusercontent.com/82097084/165747153-fb330ccd-1475-4c40-a244-429c0c722a7f.png)

For the occupation the last 5 categories were grouped under one:
-	Handlers – Cleaners
-	Farming – fishing
-	Tech support
-	Protective Services
-	Private House Service
-	Armed Forces

*Frequency per Relationship Category*

![image](https://user-images.githubusercontent.com/82097084/165748965-14569e90-a29c-4835-9b6a-ac5470a2404f.png)

For “Marital Status” only the category “Married” remained all the rest of the categories were grouped under one category “other”.

*Frequency per Race Category*

![image](https://user-images.githubusercontent.com/82097084/165749041-6d764a7b-aff7-4fed-b97c-3f2d8f864799.png)

For Race only the category “White” remained all the rest of the categories were grouped under one category “other”.

*Frequency per Workclass Category*

![image](https://user-images.githubusercontent.com/82097084/165749091-a18920ae-511f-454b-a8e7-cb8c05a58a24.png)

For “Workclass” only the category “Private” remained all the rest of the categories were grouped under one category “other”.

Because sex and class had only two categories each only a single remained (Female was dropped and Class_<50K was also dropped).
The result was that the dimensions of the new dataset decreased significantly from (9245,105) to (9245, 33). 
As a result, 92 columns were removed from the dataset.

**Normalizing the dataset**

Because of the nature of the dataset there were some variables with high values (ex. Final Weight, Capital Gain) that could be even in thousands while all the categorical values had been transformed to be either 0 or 1. 
That is why a normalization of the dataset was performed on the numerical part of the dataset in order to make all numerical values have a range between zero and one (0 - 1).
To achieve that the dataset was split into two parts.
-	Df_categorical
-	Df_numerical

Df_categorical was a dataframe that consisted of all the categorical variables after using the one hot encoder
Df_numerical on the other hand consisted of all the numerical variables (columns). The normalization was performed only on Df_numerical and not on Df_categorical
after that all the variables in the new Df_numerical after performing the normalization had values between 0 – 1. 
The new Df_numerical and the Df_categorical were concatenated (axis = 1, was used as a hyperparameter in order to concatenate the two dataframes horizontally).
The same exactly methodolofgy was followed for the Test dataset as well. The new dataframes had the final form that was needed for the ML models to be trained. 

They were saved in the local folders with the following names: 
-	“Processed_Train_Data.csv”
-	“Processed_Test_Data.csv”

**Unlabeled Dataset**

The unlabeled-project.data were used for the Semi-Supervised solution. 
The unlabeled dataset was processed by following the same methodology described above for the “Processed_Train_Data.csv”. 
The new processed unlabeled dataset was saved locally with the name.
-	Processed_Unlabeled_Data.csv
The unlabeled-project.data dataset had a difference with regards to the train-project.data and the test-project.data datasets. 
That difference was that the column “Native_Country” misses from the unlabeled-project.data. As a result, 
for the three datasets to have the same exactly columns for the analysis, the “Native_Country” column was removed from the “Processed_Train_Data.csv” and the “Processed_Test_Data.csv”. The two new datasets were named.
-	Processed_Test_Data_Without_Countries.csv
-	Processed_Train_Data_Without_Countries.csv

**Model development**

For training the models 6 different model types were used.
-	Artificial Neural Networks
-	Decision Trees
-	Support Vector Machine
-	Naïve Bayes
-	K - Nearest Neighbors
-	Random Forest

**GridSearchCV**

The GridSearchCV function from sklearn was used in order to try different combinations of hyperparameters for the models. 
More specifically the following combinations of hyperparameters were tried for each model.

*ANN (Artificial Neural Networks)*
-	'hidden_layer_sizes': [(10,5), (10,5,3), (15,10,5,3)]
-	'alpha': [0.0001, 0.05]
-	'activation': ['tanh', 'relu']
-	'max_iter':[200,400]

*DT (Decision Tree)*
-	'criterion': ['gini', 'entropy']
-	'max_depth': [10, 100, None]

*SVM (Support Vector Machine)*
-	'C': [10,100]
-	'kernel': ['rbf', 'linear', 'sigmoid']

*KNN(K- Nearest Neighbors)*
-	n_neighbors=range(1,11)

*NB (Naïve Bayes)*
-	No hyperparameter was chosen for NB as it is a really simple model and does not require fine tuning with regards to its hyperparameters

*RF (Random Forest)*
-	'n_estimators':[5,10,25]

**Ranking the features**

The mutual information technique was used in order to rank the features with regards to the most important ones. Mutual information technique ranked its feature based on its correlation to the target variable (in this case annual salary above or below $50K). The features have been sorted in ascending order and the results are provided below:
-	Rank 1: Education_ Some-college with corr 0.0
-	Rank 2: Occupation_ Sales with corr 0.0
-	Rank 3: Occupation_ Transport-moving with corr 0.0
-	Rank 4: Native_Country_ United-States with corr 0.0
-	Rank 5: Education_ Assoc-voc with corr 0.0003
-	Rank 6: Education_ Assoc-acdm with corr 0.0009
-	Rank 7: Occupation_ Craft-repair with corr 0.0029
-	Rank 8: Occupation_ Adm-clerical with corr 0.0037
-	Rank 9: Relationship_ Other-relative with corr 0.0044
-	Rank 10: Race_ White with corr 0.007
-	Rank 11: Occupation_ Prof-specialty with corr 0.0079
-	Rank 12: Education_ Doctorate with corr 0.0082
-	Rank 13: Workclass_ Private with corr 0.0091
-	Rank 14: Relationship_ Not-in-family with corr 0.0133
-	Rank 15: Final_Weight with corr 0.0134
-	Rank 16: Occupation_ Machine-op-inspct with corr 0.0135
-	Rank 17: Education_ HS-grad with corr 0.0138
-	Rank 18: Education_ Bachelors with corr 0.0148
-	Rank 19: Relationship_ Unmarried with corr 0.0151
-	Rank 20: Relationship_ Wife with corr 0.0173
-	Rank 21: Education_ Masters with corr 0.0174
-	Rank 22: Sex_ Male with corr 0.0226
-	Rank 23: Occupation_ Exec-managerial with corr 0.0229
-	Rank 24: Occupation_ Other-service with corr 0.0237
-	Rank 25: Relationship_ Own-child with corr 0.032
-	Rank 26: Capital_Loss with corr 0.0336
-	Rank 27: Hours_per_Week with corr 0.0424
-	Rank 28: Education_Num with corr 0.0661
-	Rank 29: Age with corr 0.0678
-	Rank 30: Capital_Gain with corr 0.0827
-	Rank 31: Relationship_ Husband with corr 0.0831
-	Rank 32: Marital_Status_ Married-civ-spouse with corr 0.1046

**Models Evaluation**

For the evaluation of the models Precision, Recall and F1  scores were chosen. 
As well as the ROC Curve is order to have a visual representation of the models accuracy. 

|Model|Best Parameters|Precision|Recall|F1|
|---------------|---------------|---------------|---------------|---------------|
ANN	activation': 'tanh',
 'alpha': 0.05,
 'hidden_layer_sizes': (15, 10, 5, 3),
 'max_iter': 200	0,75	0,82	0,77
DT	criterion': 'entropy', 
'max_depth': 10	0,73	0,8	0,74
SVM	C': 100, 
'kernel': 'rbf'	0,73	0,8	0,74
KNN	n_neighbors: 7	0,72	0,78	0,73
NB		0,73	0,79	0,74
RF	n_estimators': 25	0,74	0,8	0,75






