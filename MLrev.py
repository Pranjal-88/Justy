import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

def tellscores(modelname,real,pred,stats):
    acs=accuracy_score(real,pred)
    cf=confusion_matrix(real,pred)
    stats=pd.concat([stats,pd.DataFrame({"Model Name":modelname,"Accuracy":acs,"Correct":cf[1][1]+cf[0][0]
                                         ,"Wrong":cf[0][1]+cf[1][0]},index=[0])],ignore_index=True)
    return stats

stats=pd.DataFrame(columns=["Model Name","Accuracy","Correct","Wrong"]) 

#TODO:Data Reading
dataset=pd.read_csv('Datsets\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#TODO:Columns
# print(dataset.columns)
['Review', 'Liked']

#TODO:Coutplot for 'Liked'
# sns.countplot(x=dataset['Liked'])
# plt.show()
# print(dataset['Liked'].value_counts())
# Liked
# 1    500
# 0    500

#TODO:Adding a new column 'Length'
dataset['Length']=dataset['Review'].apply(len)
# print(dataset.head())

#TODO:Histogram for variable 'length'
# dataset['Length'].plot.hist(bins=50)
# plt.show()

#TODO:Segregating positive and negative comments
p_rvs=dataset[dataset['Liked']==1]
n_rvs=dataset[dataset['Liked']==0]
# print(p_rvs.head())

#TODO:Text Cleaning
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()

    ps=PorterStemmer()
    stop_words=stopwords.words('english')
    stop_words.remove('not')
    review=[ps.stem(word) for word in review if not word in set(stop_words)]
    review=' '.join(review)
    corpus.append(review)

#TODO:Sparse matrix
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1]

#TODO:Data Divison
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=42)

#TODO:Gaussian NB
mod1=GaussianNB()
mod1.fit(train_x,train_y)
pred1=mod1.predict(test_x)
stats=tellscores("Gaussian NB",test_y,pred1,stats)

#TODO:XGB Classifer
mod2=XGBClassifier()
mod2.fit(train_x,train_y)
pred2=mod2.predict(test_x)
stats=tellscores("XGB Classifer",test_y,pred2,stats)

#TODO:Print results
# print(stats)
#       Model Name  Accuracy Correct Wrong
# 0    Gaussian NB     0.670     134    66
# 1  XGB Classifer     0.705     141    59





