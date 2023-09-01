import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def tellscores(modelname,real,pred,cvi,stats):
    acs=accuracy_score(real,pred)
    cf=confusion_matrix(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"Accuracy":acs,"CV Accuracy":cvi.mean()*100,
                                   "CV Std":cvi.std()*100,"Correct Predictions":cf[0][0]+cf[1][1],
                                   "Wrong Predictions":cf[0][1]+cf[1][0]},index=[0])],ignore_index=True)
    return stats
    
stats=pd.DataFrame(columns=["Model Name","Accuracy","CV Accuracy","CV Std","Correct Predictions","Wrong Predictions"])

#TODO:Data reading
dataset=pd.read_csv(r'Datsets\WA_Fn-UseC_-HR-Employee-Attrition.csv')

#TODO:Printing all columns
# print(dataset.columns)
['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department','DistanceFromHome', 'Education', 'EducationField',
'EmployeeCount','EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel',
'JobRole', 'JobSatisfaction','MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked','Over18',
 'OverTime', 'PercentSalaryHike', 'PerformanceRating','RelationshipSatisfaction', 'StandardHours',
'StockOptionLevel','TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance','YearsAtCompany',
'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']

#TODO: Dropping columns that have no significance
dataset.drop(["EmployeeCount","EmployeeNumber","Over18","StandardHours"],axis=1,inplace=True)

#TODO:Countplot for variable 'Attrition'
# sns.countplot(x=dataset['Attrition'])
# plt.show()
# print(dataset["Attrition"].value_counts())
# Attrition
# No     1233
# Yes     237

#TODO:Differentiating countplots
# sns.countplot(x="Department",hue="Attrition",data=dataset)
# sns.countplot(x="JobRole",hue="Attrition",data=dataset)
# sns.countplot(x="JobSatisfaction",hue="Attrition",data=dataset)
# plt.show()

#TODO:A variable with all categorical cols dropped
c_cols=dataset.select_dtypes(include='object').keys()
num_data=dataset.drop(c_cols,axis=1)

#TODO:Heatmap
# corr_m=num_data.corr()
# sns.heatmap(corr_m,annot=True,cmap='coolwarm')
# h_corr=corr_m.index()
# plt.show()

#TODO: Hot Encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype=int)

#TODO:Changing name for target variable
dataset.rename(columns={"Attrition_Yes":"Attrition"},inplace=True)

#TODO:Data Divison
data_fts=dataset.drop("Attrition",axis=1)
data_lbs=dataset['Attrition']
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,test_size=0.2,random_state=42)

#TODO:Standardisation
inst=StandardScaler()
train_x=inst.fit_transform(train_x)
test_x=inst.transform(test_x)

#TODO:Logistic Regression
# mod1=LogisticRegression()
# mod1.fit(train_x,train_y)
# pred1=mod1.predict(test_x)
# cv1=cross_val_score(estimator=mod1,X=train_x,y=train_y,cv=5)
# stats=tellscores("Logistic Regression",test_y,pred1,cv1,stats)

#TODO:Random Forest Classifier
# mod2=RandomForestClassifier()
# mod2.fit(train_x,train_y)
# pred2=mod2.predict(test_x)
# cv2=cross_val_score(estimator=mod2,X=train_x,y=train_y,cv=5)
# stats=tellscores("Random Forest Classifier",test_y,pred2,cv2,stats)

#TODO:XGB Classifier
# mod3=XGBClassifier()
# mod3.fit(train_x,train_y)
# pred3=mod3.predict(test_x)
# cv3=cross_val_score(estimator=mod3,X=train_x,y=train_y,cv=5)
# stats=tellscores("XGB Classifier",test_y,pred3,cv3,stats)

#TODO:SVC
# mod4=SVC()
# mod4.fit(train_x,train_y)
# pred4=mod4.predict(test_x)
# cv4=cross_val_score(estimator=mod4,X=train_x,y=train_y,cv=5)
# stats=tellscores("SVC",test_y,pred4,cv4,stats)

#TODO:All results
# print(stats)
#                  Model Name  Accuracy  CV Accuracy    CV Std Correct Predictions Wrong Predictions
# 0       Logistic Regression  0.880952    86.310855  1.677042                 259                35
# 1  Random Forest Classifier  0.874150    85.459070  0.566908                 257                37
# 2            XGB Classifier  0.867347    85.288136  0.805589                 255                39
# 3                       SVC  0.897959    85.715831  1.369708                 264                30

#TODO:Hyperparameter Tuning for Logistic Regression
# parms={"penalty":["l1", "l2", "elasticnet", None],"solver":["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"],
#        "C":[0.25,0.5,0.75,1,1.25,1.5,1.75,2],"max_iter":[50,100,150,200]}
# rcv=RandomizedSearchCV(estimator=mod1,param_distributions=parms,n_iter=10,scoring='roc_auc',cv=5,verbose=3,n_jobs=-1)
# rcv.fit(train_x,train_y)
# print(rcv.best_estimator_)
# print(rcv.best_params_)
# print(rcv.score)
# {'solver': 'liblinear', 'penalty': 'l2', 'max_iter': 50, 'C': 0.25}

#TODO:Final Model
mod1=LogisticRegression(solver='liblinear',penalty='l2', max_iter=50,C=0.25)
mod1.fit(train_x,train_y)
pred1=mod1.predict(test_x)
cv1=cross_val_score(estimator=mod1,X=train_x,y=train_y,cv=5)
stats=tellscores("Logistic Regression",test_y,pred1,cv1,stats)

#TODO:Final Stats
# print(stats)
#             Model Name  Accuracy  CV Accuracy    CV Std Correct Predictions Wrong Predictions
# 0  Logistic Regression  0.894558    86.990624  1.431008                 263                31