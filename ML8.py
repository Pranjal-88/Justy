#CLUSTERING PROBLEM
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#TODO:Data Reading
dataset=pd.read_csv('Datsets\CC GENERAL.csv')

#TODO:Columns in dataset
# print(dataset.select_dtypes(include='object').columns)
['CUST_ID']
# print(dataset.select_dtypes(include=['int64','float64']).columns)
['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',   
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',   
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

#TODO:Null value check
# print(dataset.columns[dataset.isnull().any()])
['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']

#TODO:Countering null values
dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(),inplace=True)
dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(),inplace=True)

#TODO:Dropping JB columns
dataset.drop("CUST_ID",axis=1,inplace=True)

#TODO:Heatmap
# corr_m=dataset.corr()
# sns.heatmap(corr_m,annot=True,cmap='coolwarm')
# plt.show()
df=dataset.copy()

#TODO:Standardisation
ins=StandardScaler()
dataset=ins.fit_transform(dataset)

#TODO:Elbow method for finding appropriate number of clusters
# wcss=[]
# for i in range(1,20):
#     Km=KMeans(n_clusters=i,init='k-means++',n_init=10)
#     Km.fit(dataset)
#     wcss.append(Km.inertia_)
# plt.plot(range(1,20),wcss)
# plt.ylabel("Inertias")
# plt.xlabel("n_clusters")
# plt.show()

#TODO:Finalizing the model
k_mean=KMeans(n_clusters=8,init='k-means++',n_init=10)
pred=k_mean.fit_predict(dataset)
pred=pred.reshape(8950,1)

#TODO:Adding the target variable to the datasset
x=np.concatenate([df,pred],axis=1)
final_data=pd.DataFrame(data=x,columns=['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',   
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',   
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE','CLUSTER_NUMBER'])
final_data.to_csv(r'Datsets\trail.csv')


    
    




