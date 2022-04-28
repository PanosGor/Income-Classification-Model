# Income-Classification-Model
A Semi-Supervised classification model predicting if an individual's annual income is more or less than $50K

##Goal

The goal of this project is to create a ML model that predicts if a person’s salary will be above or below 50K dollars

##Preprocessing Phase

Three different datasets were provided for the classification part of this project. 
- test-project.data
- train-project.data
- unlabeled-project.data
- 
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
