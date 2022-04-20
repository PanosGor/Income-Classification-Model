import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
#Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#Metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
#Other
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import shuffle
#Under/Over Sampling and SMOTE
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler 


def imbalance_corrector(X,y,choise=1):
    """This function take as an argument X,y and returns back 
    X,y after applying one of the follwoing methods : Oversampling, undersampling, SMOTE
    CHOISE is the user's choise with regards to which method to use 
    1 == Under-sampling,
    2 == Over-sampling
    3 == SMOTE
    """
    
    sampling_method={1:{'model':RandomUnderSampler(sampling_strategy=1),'name':'Under-sampling'},
                     2:{'model':RandomOverSampler(sampling_strategy=1),'name':'Over-sampling'},
                     3:{'model':SMOTE(sampling_strategy='minority'),'name':"SMOTE-sampling"}}
    x_fin,y_fin=sampling_method[choise]['model'].fit_resample(X,y)
    print(" ")
    print("Number of items per class on Train Dataset after resampling:")
    print(y_fin.value_counts())
    print(" ")
    fig2, ax2 = plt.subplots()
    ax2.pie(y_fin.value_counts(), autopct='%.2f', labels=[0,1])
    ax2.set_title(sampling_method[choise]['name'])
    return x_fin,y_fin


#PCA Function
def PCA_Function(N_comp,X):
    """ This function takes as an input a number of principal componets
    and performs principal componetns analysis in order to reduce the dimensonality of the dataset
    in the end it returns the new transformed dataset"""
    
    pca = PCA(n_components=N_comp,random_state=1)
    X=pca.fit(X).transform(X)
    print(f'explained variance ratio (first {N_comp} components): %s'
      % str(pca.explained_variance_ratio_))
    print(" ")
    print(f"Total percentage of components explaining the data {round(sum(pca.explained_variance_ratio_),2)*100} %")
    print(" ")
    return X

def ranking_features(x_t,y_t):
    """ This function uses mutual Information Technique in order to rank all the 
    features according to their correlation to the class"""

    ranks={}
    metr=0
    information=mutual_info_classif(x_t,y_t)
    print(" ")
    print("Ranking the features with mutual Information technique printing in ascending order")
    for i,j in enumerate(x_t.columns):
        ranks[j]=information[i]
    print(" ")
    results={k: v for k, v in sorted(ranks.items(), key=lambda item: item[1])}
    for m,n in results.items():
        metr+=1
        print(f"Rank {metr} : {m} with corr {round(n,4)}")
    return results


df_train=pd.read_csv(r'Processed_Train_Data.csv')
df_train.head()
df_test=pd.read_csv(r'Processed_Test_Data.csv')
df_test.head()

print("Number of items per class on Train Dataset:")
print(df_train['Class_ >50K'].value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(df_train['Class_ >50K'].value_counts(), autopct='%.2f', labels=[0,1])
#grouped_res=df_train['Class_ >50K'].value_counts().plot.pie(autopct='%.2f')
ax1.set_title("Zero Indicates below 50K")
print(" ")


x_train = df_train.drop('Class_ >50K',axis=1)
y_train = df_train['Class_ >50K']
x_test = df_test.drop('Class_ >50K',axis=1)
y_test = df_test['Class_ >50K']


mutual_information_results=ranking_features(x_train,y_train)

"""Random Undersampling/Oversampling and SMOTE is used in order to balance out the dataset.
If you don't want to use any re-samplinmg technique then comment out the following line of code
1 == Under-sampling, 2 == Over-sampling, 3 == SMOTE"""

Sampling_Technique = 3
x_train, y_train = imbalance_corrector(x_train, y_train,Sampling_Technique)


"""If you want to perform PCA on the dataset then comment out the lines below"""
#N=20
#x_train=PCA_Function(N,x_train)
#x_test=PCA_Function(N,x_test)


scores=[]

"""Instead of creating many different model a dictionary named models_params was used in order 
to all the different hyperparameters for each model through GridSearch"""


models_params={
    'ANN':{'model': MLPClassifier(),'params':{ 'hidden_layer_sizes': [(10,5), (10,5,3), (15,10,5,3)],
                     'alpha': [0.0001, 0.05],
                     'activation': ['tanh', 'relu']}},
    'DT':{'model':DecisionTreeClassifier(),'params':{'criterion': ['gini', 'entropy'],
             'max_depth': [10, 100, None]}},
    'SVM':{'model':SVC(gamma='auto',probability=True),'params':{'C': [10,100],'kernel': ['rbf', 'linear', 'sigmoid']}},
    'KNN':{'model':KNeighborsClassifier(),'params':dict(n_neighbors=range(1,11))},
    'NB':{'model':GaussianNB(),'params':None},
    'RF':{'model':RandomForestClassifier(),'params':{'n_estimators':[5,10,25]}}
    }

for model_name,mp in models_params.items():
    if mp['params']==None:
        clf=mp['model']
        clf.fit(x_train,y_train)
        y_test_pred = clf.predict(x_test)
        pred =clf.predict_proba(x_test)
        fp,tp,threshold = roc_curve(y_test,pred[:,1])
        scores.append({'model_name':model_name,'best_params':None,
                       'Precission':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[0],2),
                       'Recall':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[1],2),
                       'F1':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[2],2),
                       'model':clf,'confusion_matrix':confusion_matrix(y_test, y_test_pred,labels=None),
                       'Roc_curve':[fp,tp],'Classification_report':classification_report(y_test,y_test_pred)
                       })
    else:
        clf=GridSearchCV(mp['model'],mp['params'],n_jobs=-1,cv=5,scoring='f1_macro')
        clf.fit(x_train,y_train)
        y_test_pred = clf.predict(x_test)
        pred =clf.predict_proba(x_test)
        fp,tp,threshold = roc_curve(y_test,pred[:,1])
        scores.append({'model_name':model_name,'best_params':clf.best_params_,
                       'Precission':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[0],2),
                       'Recall':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[1],2),
                       'F1':round(precision_recall_fscore_support(y_test,y_test_pred,average='macro')[2],2),
                       'model':clf,'confusion_matrix':confusion_matrix(y_test, y_test_pred,labels=None),
                       'Roc_curve':[fp,tp],'Classification_report':classification_report(y_test,y_test_pred)
                       })
        
df=pd.DataFrame(scores,columns=['model_name','best_params','Precission','Recall','F1'])

print("Models Evaluations")
print(df)
print(" ")
for i in scores:
    print(" ")
    print(f"{i['model_name']} Classification Report")
    print(i['Classification_report'])
    print(" ")
    print(f"{i['model_name']} Confusion Matrix")
    print(i['confusion_matrix'])    
  
fig, axs = plt.subplots(3,2,sharex=True, sharey=True, figsize=(12, 7))
fig.suptitle("ROC Curve")

axs[0,0].plot(scores[0]['Roc_curve'][0],scores[0]['Roc_curve'][1],color='blue',label=scores[0]['model_name'])
axs[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0,0].legend(loc="lower right")


axs[0,1].plot(scores[1]['Roc_curve'][0],scores[1]['Roc_curve'][1],color='red',label=scores[1]['model_name'])
axs[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0,1].legend(loc="lower right")

axs[1,0].plot(scores[2]['Roc_curve'][0],scores[2]['Roc_curve'][1],color='green',label=scores[2]['model_name'])
axs[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1,0].legend(loc="lower right")

axs[1,1].plot(scores[3]['Roc_curve'][0],scores[3]['Roc_curve'][1],color='orange',label=scores[3]['model_name'])
axs[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1,1].legend(loc="lower right")

axs[2,0].plot(scores[4]['Roc_curve'][0],scores[4]['Roc_curve'][1],color='yellow',label=scores[4]['model_name'])
axs[2,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[2,0].legend(loc="lower right")

axs[2,1].plot(scores[5]['Roc_curve'][0],scores[5]['Roc_curve'][1],color='purple',label=scores[5]['model_name'])
axs[2,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[2,1].legend(loc="lower right")


for ax in axs.flat:
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    
#df['best_params'].iloc[0]
#df['Precission']            




