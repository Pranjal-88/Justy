import pandas as pd
import seaborn as  sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

stats=pd.DataFrame(columns=["Model Name","r2 Score"])
def add(stats,real,pred,modelname):
    rs=r2_score(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"r2 Score":rs},index=[0])],ignore_index=True)
    return stats

#TODO: Data Reading
dataset=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\train.csv')

#TODO: Heatmap of null values
# sns.heatmap(dataset.isnull(),annot=True)
# plt.show()

#TODO->It is better to drop those columns where null values percent is more than 50%

#TODO:Finding all null columns
npct=(dataset.isnull().sum()/dataset.shape[0])*100
# print(npct[npct>0].keys())
# ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
# 'BsmtFinType2','Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish', 'GarageQual',
# 'GarageCond', 'PoolQC', 'Fence','MiscFeature']

#TODO:Dropping cloumns where null ismore than 50%
# print(npct[npct>50].keys())
# ['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature']
dataset.drop(['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)

#TODO:Placing mean in numerical null columns
# print(dataset.select_dtypes(include=['int64','float64']).columns[dataset.select_dtypes(include=['int64','float64']).isnull().any()])
# ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean(),inplace=True)
dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean(),inplace=True)
dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean(),inplace=True)

#TODO:Placing mode in categorical columns
# print(dataset.select_dtypes(include='object').columns[dataset.select_dtypes(include='object').isnull().any()])
# ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical', 'FireplaceQu', 'GarageType',
# 'GarageFinish', 'GarageQual','GarageCond']
dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode()[0],inplace=True)
dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0],inplace=True)
dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0],inplace=True)
dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0],inplace=True)
dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0],inplace=True)
dataset['Electrical'].fillna(dataset['Electrical'].mode()[0],inplace=True)
dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode()[0],inplace=True)
dataset['GarageType'].fillna(dataset['GarageType'].mode()[0],inplace=True)
dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode()[0],inplace=True)
dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0],inplace=True)
dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0],inplace=True)

#TODO:Processing on NUMERICAL COLUMNS ->A variable with no categorical columns
num_data=dataset.drop(dataset.select_dtypes(include='object').columns,axis=1)

#TODO:Plotting the linear relationship for all numeric variables wrt "SalePrice"
# dat2=num_data.drop('SalePrice',axis=1)
# num_data.corrwith(dataset['SalePrice']).plot.bar(grid=True)
# plt.show()

#TODO:Heatmap
# corr_m=num_data.corr()
# h_corr=corr_m.index[abs(corr_m['SalePrice'])>0.5]
# sns.heatmap(corr_m[h_corr].corr(),annot=True,cmap='coolwarm')
# plt.show()

#TODO:Plotting a displot
# sns.displot(dataset['SalePrice'],kde=True)
# plt.show()

#TODO:Hot encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype=int)

#TODO: Data Divison
d_fts=dataset.drop('SalePrice',axis=1)
d_lbs=dataset['SalePrice']
train_x,test_x,train_y,test_y=train_test_split(d_fts,d_lbs,test_size=0.2,random_state=42)

#TODO: Feature Scaling
inst=StandardScaler()
train_x=inst.fit_transform(train_x)
test_x=inst.transform(test_x)

#TODO:Linear Regression
# mod1=LinearRegression()
# mod1.fit(train_x,train_y)
# pred1=mod1.predict(test_x)
# stats=add(stats,test_y,pred1,"Linear Regression")

#TODO: Random Forest Regressor
# mod2=RandomForestRegressor()
# mod2.fit(train_x,train_y)
# pred2=mod2.predict(test_x)
# stats=add(stats,test_y,pred2,"Random Forest Regressor")

# #TODO: XGB Regressor
# mod3=XGBRegressor()
# mod3.fit(train_x,train_y)
# pred3=mod3.predict(test_x)
# stats=add(stats,test_y,pred3,"XGB Regressor")

#TODO: Priniting the final results
# print(stats)
#                 Model Name      r2 Score
# 0        Linear Regression -3.471282e+14
# 1  Random Forest Regressor  8.858607e-01
# 2            XGB Regressor  8.748548e-01

#TODO: On comparing best model-> Random Forest Regressor

#TODO: Hyperparameter Tuning
# param={"n_estimators":[100,200,500,1000,2000],
#        "criterion":["squared_error", "absolute_error", "friedman_mse", "poisson"]
#        ,"max_depth":[10,20,50,100],"min_samples_split":[1,2,5,10],"min_samples_leaf":[1,2,4],
#        "max_features":["sqrt", "log2", None]}
# rcv=RandomizedSearchCV(estimator=mod2,param_distributions=param,n_iter=10,cv=5,verbose=2,n_jobs=-1,random_state=0)
# rcv.fit(train_x,train_y)
# print(rcv.best_estimator_,rcv.best_params_,rcv.best_score_)
# RandomForestRegressor(criterion='friedman_mse', max_depth=20, max_features=None,
# min_samples_leaf=2, min_samples_split=10,n_estimators=1000)
# {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': None,
#   'max_depth': 20, 'criterion': 'friedman_mse'}
# 0.8374999978481752

#TODO:Final Model
modf=RandomForestRegressor(n_estimators=1000, min_samples_split=10, min_samples_leaf= 2,
                            max_features= None,max_depth= 20, criterion= 'friedman_mse')
modf.fit(train_x,train_y)
fpred=modf.predict(test_x)
stats=add(stats,test_y,fpred,"Final model")

#TODO:Final accuracy rose..
#     Model Name  r2 Score
# 0  Final model  0.884309

