import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def scr_calculate(modelname,real,pred,listat,cv):
    acs=accuracy_score(real,pred)
    pcs=precision_score(real,pred)
    rcs=recall_score(real,pred)
    f1=f1_score(real,pred)
    cf=confusion_matrix(real,pred)
    cva=cv.mean()*100
    cvs=cv.std()*100
    listat=pd.concat([listat,pd.DataFrame({"Model Name":modelname,"Accuracy":acs,"Precision":pcs,
                                           "Recall":rcs,"F1":f1,"CV Accuracy":cva,"CV std":cvs,"Correct":cf[0][0]+cf[1][1],"Wrong":cf[0][1]+cf[1][0]},index=[0])],
                                           ignore_index=True)
    return listat

scores=pd.DataFrame(columns=["Model Name","Accuracy","Precision","Recall","F1","CV Accuracy","CV std","Correct","Wrong"])

#TODO:Data reading
dataset=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\creditcard.csv')

#TODO:Check for null values
# print(dataset.isnull().values.any())
#False

#TODO:All columns
# print(dataset.columns)
# ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
# 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class']

#TODO:Countplot for variable 'Class'
# print(dataset['Class'].value_counts())
# 0    284315
# 1       492
# sns.countplot(x=dataset['Class'])
# plt.show()

#TODO: Correlation matrix and heatmap
corr_m=dataset.corr()
# sns.heatmap(corr_m,annot=True,cmap='coolwarm')
#Slightly Highly correlated features heatmap 
# high_corr=corr_m.index[abs(corr_m['Class'])>0.1]
# sns.heatmap(corr_m[high_corr].corr(),annot=True,cmap='coolwarm')
# plt.show()

#TODO:Linear graphing of all features wrt to target variable
# dat2=dataset.drop('Class',axis=1)
# dat2.corrwith(dataset['Class']).plot.bar()
# plt.show()

#TODO: Data Divison
data_fts=dataset.drop('Class',axis=1)
data_lbs=dataset['Class']
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,test_size=0.2,random_state=42)

#TODO:Standardisation
inst=StandardScaler()
train_x=inst.fit_transform(train_x)
test_x=inst.transform(test_x)

#TODO: Logistic Regression
mod1=LogisticRegression()
mod1.fit(train_x,train_y)
pred1=mod1.predict(test_x)
ins1=cross_val_score(estimator=mod1,X=train_x,y=train_y,cv=10)
scores=scr_calculate("Logistic Regression",test_y,pred1,scores,ins1)

#TODO:Random Forest Classifier
mod2=RandomForestClassifier()
mod2.fit(train_x,train_y)
pred2=mod2.predict(test_x)
ins2=cross_val_score(estimator=mod2,X=train_x,y=train_y,cv=10)
scores=scr_calculate("Random Forest Classifier",test_y,pred2,scores,ins2)

#TODO:XGB Classsifier
mod3=XGBClassifier()
mod3.fit(train_x,train_y)
pred3=mod3.predict(test_x)
ins3=cross_val_score(estimator=mod3,X=train_x,y=train_y,cv=10)
scores=scr_calculate("XGB Classsifier",test_y,pred3,scores,ins3)

#TODO: Final Scores
# print(scores)
#                  Model Name  Accuracy  Precision    Recall        F1  CV Accuracy    CV std Correct Wrong
# 0       Logistic Regression  0.999122   0.863636  0.581633  0.695122    99.920121  0.010349   56912    50
# 1  Random Forest Classifier  0.999561   0.974026  0.765306  0.857143    99.953916  0.009863   56937    25
# 2           XGB Classsifier  0.999614   0.987179  0.785714  0.875000    99.959622  0.008281   56940    22


