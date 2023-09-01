import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def tellscores(modelname,real,pred,stats):
    rs=r2_score(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"R2 Score":rs},index=[0])],ignore_index=True)
    return stats

stats=pd.DataFrame(columns=["Model Name","R2 Score"])
#TODO:Data Reading
d_set=pd.read_csv('Datsets\car data.csv')

#TODO:Columns
# print(d_set.columns)
['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
# print(d_set.select_dtypes(include='object').keys())
['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']

#TODO:Restructuring the dataset
d_set.drop('Car_Name',axis=1,inplace=True)
d_set['Year']=2023-d_set['Year']
d_set.rename(columns={'Year':'Years_Old'},inplace=True)
d_set.to_csv(r'Datsets\trail.csv')

#TODO:Hot Encoding
d_set=pd.get_dummies(d_set,drop_first=True,dtype=int)
d_set.to_csv(r'Datsets\trail.csv')

#TODO:Plotting linear relationship wrt Selling_Price
# data2=d_set.drop('Selling_Price',axis=1)
# data2.corrwith(d_set['Selling_Price']).plot.bar(grid=True)
# plt.show()

#TODO:Heatmap
# corr_m=d_set.corr()
# sns.heatmap(corr_m,annot=True,cmap='coolwarm')
# plt.show()

#TODO:Data Divison
d_fts=d_set.drop('Selling_Price',axis=1)
d_lbs=d_set['Selling_Price']
train_x,test_x,train_y,test_y=train_test_split(d_fts,d_lbs,test_size=0.2,random_state=42)

#TODO:Standardisation
ins=StandardScaler()
train_x=ins.fit_transform(train_x)
test_x=ins.transform(test_x)

#TODO:Linear Regression
# mod1=LinearRegression()
# mod1.fit(train_x,train_y)
# pred1=mod1.predict(test_x)
# stats=tellscores("Linear Regression",test_y,pred1,stats)

#TODO:Random Forest Regressor
# mod2=RandomForestRegressor()
# mod2.fit(train_x,train_y)
# pred2=mod2.predict(test_x)
# stats=tellscores("Random Forest Regressor",test_y,pred2,stats)

#TODO:XGBReggressor
# mod3=XGBRegressor()
# mod3.fit(train_x,train_y)
# pred3=mod3.predict(test_x)
# stats=tellscores("XGBReggressor",test_y,pred3,stats)

#TODO:SVC
# mod4=SVR()
# mod4.fit(train_x,train_y)
# pred4=mod4.predict(test_x)
# stats=tellscores("SVC",test_y,pred4,stats)

#TODO:Printing results
# print(stats)
#                 Model Name  R2 Score
# 0        Linear Regression  0.848981
# 1  Random Forest Regressor  0.961587
# 2            XGBReggressor  0.957531
# 3                      SVC  0.777744

#TODO:Hyperparameter tuning
# params={'n_estimators':[20,50,75,100,150,200],'criterion':["squared_error","absolute_error","friedman_mse","poisson"]
#         ,'min_samples_split':[1,2,3,4,5],'max_depth':[10,20,30,40,50],"max_features":["sqrt","log2",None]
#         ,'min_samples_leaf':[1,2,3,4,5]}
# rcv=RandomizedSearchCV(estimator=mod2,param_distributions=params,n_iter=10,cv=5,n_jobs=-1,
#                        verbose=3,scoring='neg_mean_absolute_error')
# rcv.fit(train_x,train_y)
# print(rcv.best_estimator_)
# print(rcv.best_params_)
# print(rcv.best_score_)
# RandomForestRegressor(criterion='absolute_error', max_depth=50,
#                       max_features=None, min_samples_leaf=2, n_estimators=75)
# {'n_estimators': 75, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None,
#   'max_depth': 50, 'criterion': 'absolute_error'}  
# -0.8238652777777778

#TODO:Final Model
final_mod=RandomForestRegressor(n_estimators=75,min_samples_split=2,min_samples_leaf=2,max_features=None,
                                max_depth=50,criterion='absolute_error')
final_mod.fit(train_x,train_y)
pred2=final_mod.predict(test_x)
stats=tellscores("Final Model",test_y,pred2,stats)

# print(stats)
#     Model Name  R2 Score
# 0  Final Model  0.939168

