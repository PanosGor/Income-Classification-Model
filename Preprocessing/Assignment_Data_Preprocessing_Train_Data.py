import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""This code preprocess the train dataset and creates a new version of a clean dataset called Processed_Train_Data.csv 
saved in the local folder thje new csv file is used for training the ML models
"""

def read_df(df_name):
    features=['To_go','Age','Workclass','Final_Weight','Education','Education_Num','Marital_Status','Occupation','Relationship',
         'Race','Sex','Capital_Gain','Capital_Loss','Hours_per_Week','Native_Country','Class'
         ]
    df=pd.read_csv(df_name,names=features, header=None,na_values=' ?')
    df.drop('To_go',axis=1,inplace=True)
    return df

def plot_column_distr(df,column):
    fig4, ax4 = plt.subplots(figsize=(25, 15))
    check_col=df[column].value_counts().sort_values(ascending=False)
    ax4.hist(check_col,bins=len(df_train['Native_Country'].unique()))
    ax4.set_xticklabels(df[column].unique(),rotation=45, rotation_mode="anchor", ha="right")
    ax4.set_title(f'{column} histogram')
    
    
df_train=read_df('train-project.data')
df_test=read_df('test-project.data')

print("Train Dataset Information")
print(df_train.info())
print(" ")
print("Train Dataset Description")
print(df_train.describe())


fig1, ax1 = plt.subplots()
Train_Data_Vis=sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,ax=ax1)
Train_Data_Vis.set_title('Train Data Nan-Values Visualisation')
#As we can see we have many Null values for Workclass, Occupation and Native_Country 


#The total number of rows for the whole Train and Test datasets
print(" ")
print(f"Number of rows for Train Data set {df_train.shape[0]}")
print(f"Number of rows for Test Data set {df_test.shape[0]}")
print(" ")

#Creating a dataframe of only rows that contain at least one null value
df_train_na=df_train[df_train.isnull().any(1)]
df_test_na=df_test[df_test.isnull().any(1)]
print(f"Number of rows that contain null values for Train Data set {df_train_na.shape[0]}")
print(f"Number of rows that contain null values for Test Data set {df_test_na.shape[0]}")
print(" ")
print(f"Percentage of null values in Train dataset {(df_train_na.shape[0]/df_train.shape[0])*100} %")
print(f"Percentage of null values in Test dataset {round((df_test_na.shape[0]/df_test.shape[0])*100,2)} %")
print(" ")
#Visualization heatmap on the data with only null values for Train dataset
fig2, ax2 = plt.subplots()
Nan_Train_Vis=sns.heatmap(df_train_na.isnull(),yticklabels=False,cbar=False,ax=ax2)
Nan_Train_Vis.set_title('Train NaN-Values in only NaN_DF_Train')



#Because the percentage of the rows that contain at least one null value is small for both Train and Test datasets
#and because these values are categorical (only identified in columns: Workclass, Occupation, Native_Country)
#it is not easy to be replaced by a value (mean,median)
#The best approach was decided to be for these rows to be removed from the dataset completely


df_train_2=df_train.dropna()
df_test_2=df_test.dropna()

#As we can see we have no Null values for Workclass, Occupation and Native_Country 
fig3, ax3 = plt.subplots()
Train_Data_Vis_2=sns.heatmap(df_train_2.isnull(),yticklabels=False,cbar=False,ax=ax3)
Train_Data_Vis_2.set_title('Train Data Visualisation after removing Nulls')


#The total number of rows for the whole Train and Test datasets
print(f"Number of rows for Train Data set after removing nulls {df_train_2.shape[0]}")
print(f"Number of rows for Test Data set after removing nulls {df_test_2.shape[0]}")

#Number of labels per categorical column
print(" ")
print("Number of labels per categorical column")
categorical_col=['Workclass','Education','Marital_Status','Occupation','Relationship','Race','Native_Country']
for cat in categorical_col:
    print(f"{cat}, {len(df_train_2[cat].unique())} labels")

ohe_df_train=pd.get_dummies(df_train_2)

#As can be seen after one hot encoding 90 extra columns were added into the df increasing the dimensions
print(" ")
print(f"df shape before One hot encoding: {df_train_2.shape}")
print(f"df shape after One hot encoding: {ohe_df_train.shape}")
print(" ")


#We can see that Native_Country, Education and Occupation have the most categories (>10)
plot_column_distr(df_train_2,"Native_Country")


#As we can see for many categories like Native_Country there are many observations under only one category
#Which means that we can take only the first N categories and remove everything else from the rest of categiries
def removing_features(df,list_of_features):
    for i in list_of_features:
        df.drop(i,axis=1,inplace=True)
    return df

countries_rmv=['Native_Country_'+str(i) for i in df_train_2['Native_Country'].unique()]
df_train_3=removing_features(ohe_df_train,countries_rmv[1:])

#Here we make a list withh all the categories of education that we want to remove
Education_rmv=list(df_train_2['Education'].value_counts().sort_values(ascending=False).index)[6:]
#Doctorate was removed from the list because it is a very significant factor to someone's salary
Education_rmv.pop(6)
Education_rmv_fnl=['Education_'+str(i) for i in Education_rmv]
df_train_3=removing_features(ohe_df_train,Education_rmv_fnl)  
 
#the majority of the data are for Race = white so we can remove the rest
Race_rmv=['Race_'+str(i) for i in df_train_2['Race'].unique()]
df_train_3=removing_features(ohe_df_train,Race_rmv[1:])   

#Droping the Female Column we have only two values for sex
df_train_3.drop('Sex_ Female',axis=1,inplace=True)

#Dropping >50k as we have only two classes
df_train_3.drop('Class_ <=50K',axis=1,inplace=True)


Occupation_rmv=list(df_train_2['Occupation'].value_counts().sort_values(ascending=False).index)[8:]
Occupation_rmv_final=['Occupation_'+str(i) for i in Occupation_rmv]
df_train_3=removing_features(ohe_df_train,Occupation_rmv_final) 

Marital_Status_rmv=list(df_train_2['Marital_Status'].value_counts().sort_values(ascending=False).index)[1:]
Marital_Status_rmv_final=['Marital_Status_'+str(i) for i in Marital_Status_rmv]
Marital_Status_rmv=list(df_train_2['Marital_Status'].value_counts().sort_values(ascending=False).index)[1:]
Marital_Status_rmv_final=['Marital_Status_'+str(i) for i in Marital_Status_rmv]

workclass=list(df_train_2['Workclass'].value_counts().sort_values(ascending=False).index)[1:]
workclass_rmv=['Workclass_'+str(i) for i in workclass]
df_train_3=removing_features(ohe_df_train,workclass_rmv)


#We ended up from 105 columns after One hoT Encoding to 32 after the modifications above removing 73 columns
print(" ")
print("Columns Left after preprocessing")
print(df_train_3.columns)
print(" ")
print("Train Dataset shape after preprocessing")
print(df_train_3.shape)
print(" ")

#Splitting the categorical from the numerical columns to perform Standard Scaler only on the numerical
df_train_3_num=df_train_3[list(df_train_3.columns[:6])]
df_train_3_cat=df_train_3[list(df_train_3.columns[6:])]

# normalizing Dataset
df_train_3_num=df_train_3_num.apply(lambda x: (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))
final_train_df=pd.concat([df_train_3_num,df_train_3_cat],axis=1)
final_train_df.to_csv('Processed_Train_Data.csv',index=False)