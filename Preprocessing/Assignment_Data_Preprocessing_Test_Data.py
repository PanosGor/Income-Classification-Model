import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""This code preprocess the test dataset and creates a new version of a clean dataset called Processed_Test_Data.csv 
saved in the local folder thje new csv file is used for evaluating the ML models
"""

def read_df(df_name):
    features=['To_go','Age','Workclass','Final_Weight','Education','Education_Num','Marital_Status','Occupation','Relationship',
         'Race','Sex','Capital_Gain','Capital_Loss','Hours_per_Week','Native_Country','Class'
         ]
    df=pd.read_csv(df_name,names=features, header=None,na_values=' ?')
    df.drop('To_go',axis=1,inplace=True)
    return df

df_test=read_df('test-project.data')

Test_Data_Vis=sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False)
Test_Data_Vis.set_title('Test Data Visualisation')

df_test_na=df_test[df_test.isnull().any(1)]
print(" ")
print(f"Number of rows that contain null values for Test Data set {df_test_na.shape[0]}")
print(f"Percentage of null values in Test dataset {round((df_test_na.shape[0]/df_test.shape[0])*100,2)} %")

sns.heatmap(df_test_na.isnull(),yticklabels=False,cbar=False)

df_test_2=df_test.dropna()

Test_Data_Vis_2=sns.heatmap(df_test_2.isnull(),yticklabels=False,cbar=False)
Test_Data_Vis_2.set_title('Test Data Visualisation after removing Nulls')

print(" ")
print(f"Number of rows for Test Data set after removing nulls {df_test_2.shape[0]}")
print(" ")

#Number of labels per categorical column
categorical_col=['Workclass','Education','Marital_Status','Occupation','Relationship','Race','Native_Country']
for cat in categorical_col:
    print(f"{cat}, {len(df_test_2[cat].unique())} labels")
    
ohe_df_test=pd.get_dummies(df_test_2)
print(" ")
print(f"df shape before One hot encoding: {df_test_2.shape}")
print(f"df shape after One hot encoding: {ohe_df_test.shape}")
print(" ")

#These are the columns that were left after preprocessing for the Train Dataset
#In order for this analysis to be meaningful both Train and Test Datasets 
#Need to have exactly the same columns as a results we kept the only the columns from the Train dataset 
final_cols=['Age', 'Final_Weight', 'Education_Num', 'Capital_Gain', 'Capital_Loss',
       'Hours_per_Week', 'Workclass_ Private', 'Education_ Assoc-acdm',
       'Education_ Assoc-voc', 'Education_ Bachelors', 'Education_ Doctorate',
       'Education_ HS-grad', 'Education_ Masters', 'Education_ Some-college',
       'Marital_Status_ Married-civ-spouse', 'Occupation_ Adm-clerical',
       'Occupation_ Craft-repair', 'Occupation_ Exec-managerial',
       'Occupation_ Machine-op-inspct', 'Occupation_ Other-service',
       'Occupation_ Prof-specialty', 'Occupation_ Sales',
       'Occupation_ Transport-moving', 'Relationship_ Husband',
       'Relationship_ Not-in-family', 'Relationship_ Other-relative',
       'Relationship_ Own-child', 'Relationship_ Unmarried',
       'Relationship_ Wife', 'Race_ White', 'Sex_ Male',
       'Native_Country_ United-States', 'Class_ >50K']

final_df_test=ohe_df_test[final_cols]

#Splitting the categorical from the numerical columns to perform Standard Scaler only on the numerical
df_test_num=final_df_test[list(final_df_test.columns[:6])]
df_test_cat=final_df_test[list(final_df_test.columns[6:])]

print(" ")
print("Numerical Data")
print(df_test_num.head())
print(" ")
print("Categorical Data")
print(df_test_cat.head())
print(" ")

# normalizing Dataset
df_test_num=df_test_num.apply(lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))

final_test_df=pd.concat([df_test_num,df_test_cat],axis=1)

#saving down the new clean test dataset
final_test_df.to_csv('Processed_Test_Data.csv',index=False)


