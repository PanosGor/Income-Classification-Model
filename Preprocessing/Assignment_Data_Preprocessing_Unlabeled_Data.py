import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""This script uses one hot encoder to create a new Train.csv file for the semi-supervised solution
The reason is that in the unlabeled dataset we don't have any data for the countries
As a result the country column need to be removed from the training dataset as well as the test dataset
in order the unlabeled dataset and training/testing dataset to have the same exact columns
"""

def read_df(df_name):
    features=['To_go','Age','Workclass','Final_Weight','Education','Education_Num','Marital_Status','Occupation','Relationship',
         'Race','Sex','Capital_Gain','Capital_Loss','Hours_per_Week','Native_Country','Class'
         ]
    df=pd.read_csv(df_name,names=features, header=None,na_values=' ?')
    df.drop('To_go',axis=1,inplace=True)
    return df

df_train = read_df('train-project.data')
df_test = read_df('test-project.data')
df_unlabeled = read_df('unlabeled_project.data')

ohe_df_train = pd.get_dummies(df_train)
ohe_df_test = pd.get_dummies(df_test)
ohe_df_unlabeled = pd.get_dummies(df_unlabeled)


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
       'Relationship_ Wife', 'Race_ White', 'Sex_ Male', 'Class_ >50K']

# the columns Class_ >50K has been excluded for the unlabeled data as it does not exist there
final_cols_unlabeled=['Age', 'Final_Weight', 'Education_Num', 'Capital_Gain', 'Capital_Loss',
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
       'Relationship_ Wife', 'Race_ White', 'Sex_ Male']

#Use one hot encoding to encode categorical columns
final_df_train = ohe_df_train[final_cols]
final_df_test = ohe_df_test[final_cols]
final_df_unlabeled = ohe_df_unlabeled[final_cols_unlabeled]


#Splitting the categorical from the numerical columns for the train dataset to perform Standard Scaler only on the numerical
df_train_num=final_df_train[list(final_df_train.columns[:6])]
df_train_cat=final_df_train[list(final_df_train.columns[6:])]

#Splitting the categorical from the numerical columns for the Test dataset to perform Standard Scaler only on the numerical
df_test_num=final_df_test[list(final_df_test.columns[:6])]
df_test_cat=final_df_test[list(final_df_test.columns[6:])]

#Splitting the categorical from the numerical columns for the Unlabaled dataset to perform Standard Scaler only on the numerical
df_unlabeled_num=final_df_unlabeled[list(final_df_unlabeled.columns[:6])]
df_unlabeled_cat=final_df_unlabeled[list(final_df_unlabeled.columns[6:])]

# normalizing Train Dataset
df_train_num=df_train_num.apply(lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))

# normalizing Test Dataset
df_test_num=df_test_num.apply(lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))

# normalizing Unlabeled Dataset
df_unlabeled_num=df_unlabeled_num.apply(lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))

final_train_df=pd.concat([df_train_num,df_train_cat],axis=1)
final_test_df=pd.concat([df_test_num,df_test_cat],axis=1)
final_unlabeled_df=pd.concat([df_unlabeled_num,df_unlabeled_cat],axis=1)


#Saving the processed csv files without the column Country to the local folder
final_train_df.to_csv('Processed_Train_Data_Without_Countries.csv',index=False)
final_test_df.to_csv('Processed_Test_Data_Without_Countries.csv',index=False)
final_unlabeled_df.to_csv('Processed_Unlabeled_Data.csv',index=False)