import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

Results=pd.DataFrame(columns=["Modelname","Accuracy","F1","Recall score","Precision"])
CV_results=pd.DataFrame(columns=["Modelname","CV accuracy","CV std"])

#TODO:Function for telling scores
def tellscore(modelname,x,y):
    ac=accuracy_score(x,y)
    f1=f1_score(x,y)
    rs=recall_score(x,y)
    ps=precision_score(x,y)
    dic=pd.DataFrame({"Modelname":modelname,"Accuracy":ac,"F1":f1,"Recall score":rs,"Precision":ps},index=[0])
    return dic

#TODO: Data Reading
churn_data=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\Pro2.csv')
# Columns of dataset:['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 
#  'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 
# 'EstimatedSalary', 'Exited']
# COLUMNS -> Rownumber,CustomerId , Surname play no role here

#TODO: Dropping specific columns
churn_data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1,inplace=True)

#TODO: Hot encoding
churn_data=pd.get_dummies(data=churn_data,drop_first=True,dtype=int)
# churn_data.to_csv('trail.csv')

#TODO: Countplot for target variable
# sns.countplot(x=churn_data['Exited'])
# print(churn_data['Exited'].value_counts())
# plt.show()

#TODO: Linear realtionship wrt Exited
corr_m=churn_data.corr()
# print(corr_m['Exited'].sort_values(ascending=False))

#TODO:To plot the linear relationship between all the variables
# data2=churn_data.drop('Exited',axis=1)
# data2.corrwith(churn_data['Exited']).plot.bar(title="PLOT")
# plt.show()

#TODO: Heatmap between variables
# sns.heatmap(corr_m,annot=True)
# plt.show()

#TODO: Splitting the dataset
data_x=churn_data.drop('Exited',axis=1)
data_y=churn_data['Exited']
train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.2,random_state=42)

#TODO:Feature Scaling
ins=StandardScaler()
train_x=ins.fit_transform(train_x)
test_x=ins.transform(test_x)

#TODO: Modelling a Logistic Regressor
# mod=linear_model.LogisticRegression()
# mod.fit(train_x,train_y)
# LG_predicted=mod.predict(test_x)
# #Confusion Matrix
# # print("Confusion matrix:\n",confusion_matrix(test_y,LG_predicted))
# Results=pd.concat([Results,tellscore("Logistic Regressor",test_y,LG_predicted)],ignore_index=True)
# #Cross validation score
# cvs=cross_val_score(estimator=mod,X=train_x,y=train_y,cv=10)
# CV_results=pd.concat([CV_results,pd.DataFrame({"Modelname":"Logistic Regressor","CV accuracy":cvs.mean()*100,"CV std":cvs.std()*100},index=[0])])

#TODO: Modelling a Random Forest Classifier
# rfc=RandomForestClassifier()
# rfc.fit(train_x,train_y)
# RFC_predicted=rfc.predict(test_x)
# #Confusion matrix
# # print("\nConfusion Matrix:\n ",confusion_matrix(test_y,RFC_predicted))
# Results=pd.concat([Results,tellscore("Random Forest Classifier",test_y,RFC_predicted)],ignore_index=True)
# #Cross validation Score
# cvs1=cross_val_score(estimator=rfc,X=train_x,y=train_y,cv=10)
# CV_results=pd.concat([CV_results,pd.DataFrame({"Modelname":"Random Forest Classifier","CV accuracy":cvs1.mean()*100,"CV std":cvs1.std()*100},index=[0])])

#TODO: Modelling a XGBoost Classifier
# xgb=XGBClassifier()
# xgb.fit(train_x,train_y)
# xgb_predict=xgb.predict(test_x)
# #Confusion Matrix
# # print(confusion_matrix(test_y,xgb_predict))
# Results=pd.concat([Results,tellscore("XG Boost Classifier",test_y,xgb_predict)],ignore_index=True)
# #Cross validation score
# cvs2=cross_val_score(estimator=xgb,X=train_x,y=train_y,cv=10)
# CV_results=pd.concat([CV_results,pd.DataFrame({"Modelname":"XG Boost Classifier","CV accuracy":cvs2.mean()*100,"CV std":cvs2.std()*100},index=[0])])

#TODO: print(Results)
#                   Modelname  Accuracy        F1  Recall score  Precision
# 0        Logistic Regressor    0.8110  0.294776      0.201018   0.552448
# 1  Random Forest Classifier    0.8685  0.592248      0.486005   0.757937
# 2       XG Boost Classifier    0.8580  0.577381      0.493639   0.695341

#TODO: print(CV_Results)
#                   Modelname  CV accuracy    CV std
# 0        Logistic Regressor       80.875  1.101136
# 0  Random Forest Classifier       86.075  0.512957
# 0       XG Boost Classifier       85.200  0.995615

#TODO: Clearly the best model here is Random Forest Classifier
#TODO: Hyperparameter Tuning for Random Forest Classifier
# para={'criterion':["gini", "entropy", "log_loss"],'max_features':["sqrt", "log2", None],'n_estimators':[10,50,100,200,500]}
# rcv=RandomizedSearchCV(estimator=rfc,param_distributions=para,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# rcv.fit(train_x,train_y)
# print(rcv.best_estimator_,rcv.best_score_,rcv.best_params_)
#RandomForestClassifier(criterion='log_loss', n_estimators=200) 0.8541212320368464 
# {'n_estimators': 200, 'max_features': 'sqrt', 'criterion': 'log_loss'}

#TODO: Finalizing the Random Forest Classifier Model
rfc=RandomForestClassifier(n_estimators= 200, max_features= 'sqrt', criterion= 'log_loss') 
rfc.fit(train_x,train_y)
final_pred=rfc.predict(test_x)
# print(tellscore("Final Model",test_y,final_pred))
#     Modelname  Accuracy        F1  Recall score  Precision
# 0  Final Model    0.8675  0.583987      0.473282   0.762295 
cvs=cross_val_score(estimator=rfc,X=train_x,y=train_y,cv=10)
# print(f"Accuracy:{cvs.mean()*100}\n Std:{cvs.std()*100}")
# Accuracy:86.4125       
# Std:0.6448110188264465





