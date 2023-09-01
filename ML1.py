import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

def tellresults(modelname,real,predicted):
    ac_s=accuracy_score(real,predicted)
    pc_s=precision_score(real,predicted)
    f1_s=f1_score(real,predicted)
    rc_s=recall_score(real,predicted)
    df=pd.DataFrame({"Modelname":modelname,"Accuracy score":ac_s,"F1 score":f1_s,"Precision score":pc_s,"Recall score":rc_s},index=[0])
    return(df)

final=pd.DataFrame(columns=['Modelname','Accuracy score','F1 score','Precision score','Recall score'])

#TODO:Data reading
data_BC=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\Bcdata.csv')
data_BC=pd.get_dummies(data=data_BC,drop_first=True,dtype=int)

#TODO:To draw the linear relationship in the form of bar graphs
# bc_fts=data_BC.drop('diagnosis_M',axis=1)
# bc_fts.corrwith(data_BC['diagnosis_M']).plot.bar()
# plt.show()

#TODO:To draw the heatmap between variables
# corr_m=data_BC.corr()
# sns.heatmap(corr_m,annot=True)
# plt.show()

#TODO:for studying the linear relationship wrt to the label
# corr_m=data_BC.corr()
# print(corr_m['diagnosis_M'].sort_values(ascending=False))

#TODO:Plots value graph for 0 and 1 (bar graph) 
# sns.countplot(x=data_BC['diagnosis_M'])
# plt.show()

#TODO: Dividing the data into train and test frames
bc_fts=data_BC.drop('diagnosis_M',axis=1)
bc_fts=bc_fts.drop('id',axis=1)
bc_lab=data_BC['diagnosis_M']

train_x,test_x,train_y,test_y=train_test_split(bc_fts,bc_lab,test_size=0.2,random_state=42)

#TODO:Standardisation/Normalisation
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

#TODO: Logistics Regressor
mod=linear_model.LogisticRegression()
mod.fit(train_x,train_y)
test_p=mod.predict(test_x)
final=pd.concat([final,tellresults("Logistics Regressor",test_p,test_y)])
# CROSS VALUED ERROR
acc=cross_val_score(estimator=mod,X=train_x,y=train_y,cv=10)
# print(f"Acccuracy:{acc.mean()*100}\n Standard Deviation:{acc.std()*100}\n")
# To print a confusion matrix
# print(confusion_matrix(test_y,test_p))

# TODO: Random Forest Classifier
cls=RandomForestClassifier(random_state=42)
cls.fit(train_x,train_y)
test_r=cls.predict(test_x)
final=pd.concat([final,tellresults("Random Forest Classifier",test_r,test_y)])
#CROSS VALUED ERROR
acc1=cross_val_score(estimator=cls,X=train_x,y=train_y,cv=10)
# print(f"Acccuracy:{acc1.mean()*100}\n Standard Deviation:{acc1.std()*100}\n")
#To print a confusion matrix
# print(confusion_matrix(test_y,test_r))

#TODO: Clearlly Logistic Regressor >>>>>> Random Forest Classsifier
#TODO: Hyperparameter tuning and designing the final Logistic Regressor model using RandomisedSearchCV
# para={'penalty':['l1', 'l2', 'elasticnet', None],'C':[0.25,0.5,0.75,1,1.25,1.5,1,75,2],'max_iter':[100,200,500,700,1000],'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
# rcv=RandomizedSearchCV(estimator=mod,param_distributions=para,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
# rcv.fit(train_x,train_y)
# print("THIS->",rcv.best_estimator_,rcv.best_score_,rcv.best_params_)
# The output shows: LogisticRegression(C=1, max_iter=200, solver='newton-cg') , 0.992618081715445,{'solver': 'sag', 'penalty': 'l2', 'max_iter': 500, 'C': 0.5}

#TODO: Finalizing the Logistic Regressor model
mod_new=linear_model.LogisticRegression(C=0.5, max_iter=700, solver='newton-cg')
mod_new.fit(train_x,train_y)
final_pred=mod_new.predict(test_x)
final=pd.concat([final,tellresults("Final Model",test_y,final_pred)])
#Cross Validation
# cv=cross_val_score(estimator=mod_new,X=train_x,y=train_y,cv=10)
# print(f"Acccuracy:{cv.mean()*100}\n Standard Deviation:{cv.std()*100}\n")

print(final)






