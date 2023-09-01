import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')   
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#TODO:Data Reading
dataset=pd.read_csv('Datsets\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#TODO:Columns
# print(dataset.columns)
['Review', 'Liked']

#TODO:Countplot for 'Liked'
# sns.countplot(x=dataset['Liked'])
# plt.show()
# print(dataset['Liked'].value_counts())
# Liked
# 1    500
# 0    500

#TODO:Adding a new column length
# dataset['Length_Review']=dataset['Review'].apply(len)
# print(dataset.head())

#TODO:Histogram for 'Length'
# dataset['Length_Review'].plot.hist(bins=50)
# plt.show()
# print(dataset['Length_Review'].describe())
# count    1000.000000
# mean       58.315000
# std        32.360052
# min        11.000000
# 25%        33.000000
# 50%        51.000000
# 75%        80.000000
# max       149.000000

#TODO:Segregation on basis of 'Liked'
p_rvs=dataset[dataset['Liked']==1]
n_rvs=dataset[dataset['Liked']==0]

#TODO:Text Cleaning
corpus=[]
for i in range(0,len(dataset)):
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
y=dataset['Liked']

#TODO:Data Divison:
# train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=42)

#TODO:XGBClassifier
xg=XGBClassifier()
xg.fit(x,y)
# print(test_x)
# pred1=xg.predict(test_x)
# print(pred1)
# print("Accuracy:",accuracy_score(test_y,pred1))

#TODO: Data Input
corpus=[]
x=input("Enter review:")
rev=re.sub('[^A-Za-z]',' ',x)
rev=rev.lower()
rev=rev.split()
test_x=[ps.stem(word) for word in rev if not word in set(stop_words)]
test_x=' '.join(test_x)
corpus.append(test_x)

test_x=cv.transform(corpus).toarray()
pred=xg.predict(test_x)

if (pred[0]==1):
    print("Positive")
else:
    print("Negative")


