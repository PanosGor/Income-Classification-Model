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
•	Age
•	Final_Weight
•	Education_Num
•	Capital_Gain
•	Capital_Loss
•	Hours_per_week

*Nominal*
•	WorkClass
•	Education
•	Marital_Status
•	Occupation
•	Relationship
•	Race
•	Sex
•	Native_Country
•	Class
