import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


results=pd.DataFrame(columns=["Model Name","r2 Score"])
def add(modelname,real,pred,results):
    r2s=r2_score(real,pred)
    results=pd.concat([results,pd.DataFrame({"Model Name":modelname,"r2 Score":r2s},index=[0])],ignore_index=True)
    return results

#TODO: Data Reading
dataset=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\insurance.csv')

#TODO: Hot encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype=int)
# print(dataset.columns)
# ['age', 'bmi', 'children', 'charges', 'sex_male', 'smoker_yes',
# 'region_northwest', 'region_southeast', 'region_southwest']

#TODO: To plot linear relationship wrt to charges
# dataset1=dataset.drop('charges',axis=1)
# dataset1.corrwith(dataset['charges']).plot.bar(grid=True)
# plt.show()

#TODO: To draw a heatmap
corr_m=dataset.corr()
sns.heatmap(corr_m,annot=True)
plt.show()

#TODO: Splitting the dataset
data_fts=dataset.drop('charges',axis=1)
data_lbs=dataset['charges']
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,test_size=0.2,random_state=42)

#TODO: Feature Scaling
inst=StandardScaler()
train_x=inst.fit_transform(train_x)
test_x=inst.transform(test_x)

# #TODO:Linear Regression model
# mod=LinearRegression()
# mod.fit(train_x,train_y)
# LR_pred=mod.predict(test_x)
# results=add("Linear Regression",test_y,LR_pred,results)

#TODO:Random Forest Regressor
rcf=RandomForestRegressor()
rcf.fit(train_x,train_y)
RFC_pred=rcf.predict(test_x)
results=add("Random Forest Regressor",test_y,RFC_pred,results)

# #TODO: XGB Regressor
# xgb=XGBRegressor()
# xgb.fit(train_x,train_y)
# XGB_pred=xgb.predict(test_x)
# results=add("XGB Regressor",test_y,XGB_pred,results)

#TODO: Final Results
# print(results)
#                 Model Name  r2 Score
# 0        Linear Regression  0.783593
# 1  Random Forest Regressor  0.862671
# 2            XGB Regressor  0.833167
 
#TODO: Final model->>> Random Forest Regressor 

