import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def compute(stats,modelname,real,pred):
    acs=accuracy_score(real,pred)
    pcs=precision_score(real,pred)
    fs=f1_score(real,pred)
    rs=recall_score(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"Accuracy":acs,"Precision":pcs,"F1":fs,"Recall":rs},index=[0])],ignore_index=True)
    return stats

def cvcompute(stats,modelname,real,pred,cv_mean,cv_std):
    cf=confusion_matrix(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"CV Mean":cv_mean,"CV Std":cv_std,
                                         "Wrong Predictions":cf[0][1]+cf[1][0],
                                         "Correct Predictions":cf[0][0]+cf[1][1]},index=[0])],ignore_index=True)
    return stats

stats=pd.DataFrame(columns=["Model Name","Accuracy","Precision","F1","Recall"])
stats2=pd.DataFrame(columns=["Model Name","CV Mean","CV Std","Correct Predictions","Wrong Predictions"])

#TODO:Data Reading
dataset=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\Financial-Data.csv')

#TODO:Hot Encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype=int)

#TODO:Countplot for target variable
# sns.countplot(x=dataset['e_signed'])
# plt.show()
# e_signed
# 1    9639
# 0    8269

#TODO:Plotting a linear graph comparing all entities with e_signed
# datasetc=dataset.drop('e_signed',axis=1)
# datasetc.corrwith(dataset['e_signed']).plot.bar()
# plt.show()

#TODO:Finding all correlations(+ve and -ve)
corr_m=dataset.corr()
p_corr=corr_m.columns[corr_m['e_signed']>0]
['Entry_id', 'income', 'months_employed', 'years_employed', 'has_debt',
'amount_requested', 'risk_score', 'risk_score_2', 'risk_score_4',     
'e_signed', 'pay_schedule_semi-monthly', 'pay_schedule_weekly']
n_corr=corr_m.columns[corr_m['e_signed']<=0]
['age', 'home_owner', 'current_address_year', 'personal_account_m', 
'personal_account_y', 'risk_score_3', 'risk_score_5','ext_quality_score',
 'ext_quality_score_2', 'inquiries_last_month','pay_schedule_monthly']

#TODO:Restructuring the dataset
dataset['months_employed']=dataset['months_employed']+dataset['years_employed']*12
dataset.drop('years_employed',axis=1,inplace=True)
dataset['personal_account_m']=dataset['personal_account_m']+dataset['personal_account_y']*12
dataset.drop('personal_account_y',axis=1,inplace=True)

#TODO:Dropping useless columns
dataset.drop('Entry_id',axis=1,inplace=True)

#TODO:Heatmap for variables
# corr_m=dataset.corr()
# sns.heatmap(corr_m,annot=True,cmap='coolwarm')
# plt.show()

#TODO:Data Divison
data_fts=dataset.drop('e_signed',axis=1)
data_lbs=dataset['e_signed']
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,test_size=0.2,random_state=42)

#TODO:Feature Scaling
ss=StandardScaler()
train_x=ss.fit_transform(train_x)
test_x=ss.transform(test_x)

#TODO:Logistic Regressor
# mod1=LogisticRegression()
# mod1.fit(train_x,train_y)
# pred1=mod1.predict(test_x)
# stats=compute(stats,"Logistic Regressor",test_y,pred1)
# x=cross_val_score(estimator=mod1,X=train_x,y=train_y,cv=5)
# stats2=cvcompute(stats2,"Logistic Regressor",test_y,pred1,x.mean()*100,x.std()*100)

#TODO:Random Forest Classifier
# mod2=RandomForestClassifier()
# mod2.fit(train_x,train_y)
# pred2=mod2.predict(test_x)
# stats=compute(stats,"Random Forest Classifier",test_y,pred2)
# y=cross_val_score(estimator=mod2,X=train_x,y=train_y,cv=5)
# stats2=cvcompute(stats2,"Random Forest Classifier",test_y,pred2,y.mean()*100,y.std()*100)

#TODO:XGB Classifier
# mod3=XGBClassifier()
# mod3.fit(train_x,train_y)
# pred3=mod3.predict(test_x)
# stats=compute(stats,"XGB Classifier",test_y,pred3)
# z=cross_val_score(estimator=mod3,X=train_x,y=train_y,cv=5)
# stats2=cvcompute(stats2,"XGB Classifier",test_y,pred3,z.mean()*100,z.std()*100)

#TODO:Supervised Machine Classifier
# mod4=SVC()
# mod4.fit(train_x,train_y)
# pred4=mod4.predict(test_x)
# stats=compute(stats,"Supervised Machine Classifier",test_y,pred4)
# q=cross_val_score(estimator=mod4,X=train_x,y=train_y,cv=5)
# stats2=cvcompute(stats2,"Supervised Machine Classifier",test_y,pred4,q.mean()*100,q.std()*100)

#TODO:Printing stats
# print(stats)
#                       Model Name  Accuracy  Precision        F1    Recall
# 0             Logistic Regressor  0.567560   0.581585  0.636811  0.703627
# 1       Random Forest Classifier  0.626745   0.647586  0.660574  0.674093
# 2                 XGB Classifier  0.633724   0.652067  0.668854  0.686528
# 3  Supervised Machine Classifier  0.609715   0.620145  0.662645  0.711399

#TODO:Printing stats2
# print(stats2)
#                       Model Name    CV Mean    CV Std Correct Predictions Wrong Predictions
# 0             Logistic Regressor  57.594578  0.386107                2033              1549
# 1       Random Forest Classifier  62.452873  0.313653                2245              1337
# 2                 XGB Classifier  62.501758  0.729334                2270              1312
# 3  Supervised Machine Classifier  60.540369  0.685045                2184              1398

#TODO: Hyperparameter tuning for Random Forest Classifier
# parms={"learning_rate":[0.05,0.1,0.15,0.2,0.25,0.3],"max_depth":[2,4,6,8,10,12,14],"min_child_weight":[1,3,5,7]
#        ,"gamma":[0.0,0.1,0.2,0.3,0.4],"colsample_bytree":[0.3,0.4,0.5,0.6,0.7],"n_estimators":[100,200,300,400,500],
#        "subsample":[0.5,0.7,1]}
# rcv=RandomizedSearchCV(estimator=mod3,param_distributions=parms,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# rcv.fit(train_x,train_y)
# print(rcv.best_estimator_,rcv.best_score_,rcv.best_params_)
# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=0.3, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=0.0, gpu_id=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.05, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=10, max_leaves=None,
#               min_child_weight=7, missing=nan, monotone_constraints=None,
#               n_estimators=300, n_jobs=None, num_parallel_tree=None,
#               predictor=None, random_state=None, ...) 
# 0.6868255105105339 
# {'subsample': 0.7, 'n_estimators': 300, 'min_child_weight': 7, 'max_depth': 10, 'learning_rate': 0.05, 'gamma': 0.0, 'colsample_bytree': 0.3}

#TODO:Building the final model
finalmod=XGBClassifier(subsample= 0.7,n_estimators=300,min_child_weight=7,
                        max_depth=10, learning_rate=0.05,gamma=0.0,colsample_bytree=0.3)
finalmod.fit(train_x,train_y)
predf=finalmod.predict(test_x)
stats=compute(stats,"Final Model",test_y,predf)
cva=cross_val_score(estimator=finalmod,X=train_x,y=train_y,cv=10)
stats2=cvcompute(stats2,"Final Model",test_y,predf,cva.mean(),cva.std())

#TODO:Final stats
# print(stats)
#     Model Name  Accuracy  Precision        F1    Recall
# 0  Final Model  0.633445   0.651448  0.669019  0.687565
# print(stats2)
#     Model Name   CV Mean    CV Std Correct Predictions Wrong Predictions
# 0  Final Model  0.638351  0.014405                2269              1313
