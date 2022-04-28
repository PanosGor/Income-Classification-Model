# Income-Classification-Model
A Semi-Supervised classification model predicting if an individual's annual income is more or less than $50K

## Goal

The goal of this project is to create a ML model that predicts if a person’s salary will be above or below 50K dollars

## Preprocessing Phase

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






