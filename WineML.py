import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

#TODO:Data Reading
dataset=pd.read_csv('Datsets\wine-clustering.csv')

#TODO:Columns
# print(dataset.columns)
['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',     
'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']

#TODO:Plotting heatmap
# corr_m=dataset.corr()
# sns.heatmap(corr_m,annot=True)
# plt.show()
df=dataset.copy()

#TODO:Standardisation
inst=StandardScaler()
dataset=inst.fit_transform(dataset)

#TODO:Elbow method
# wcss=[]
# for i in range(1,20):
#     km=KMeans(n_clusters=i,init='k-means++',n_init=10)
#     km.fit(dataset)
#     wcss.append(km.inertia_)
# plt.plot(range(1,20),wcss)
# plt.show()

#TODO:Model Finalization
mod=KMeans(n_clusters=3,init='k-means++',n_init=10)
pred=mod.fit_predict(dataset)
pred=pred.reshape(len(pred),1)

#TODO:Putting the data
b=np.concatenate([df,pred],axis=1)
final_data=pd.DataFrame(data=b,columns=['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',     
'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline','Cluster_Number'])
final_data.to_csv('trail.csv')




