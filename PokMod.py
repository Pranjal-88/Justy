import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

#TODO:Data Extraction
dataset=pd.read_csv(r'Datsets\pokemon.csv')
# print(dataset.shape)
# (801, 41)

#TODO:Columns
# print(dataset.columns)
['abilities', 'against_bug', 'against_dark', 'against_dragon',        
'against_electric', 'against_fairy', 'against_fight', 'against_fire',
'against_flying', 'against_ghost', 'against_grass', 'against_ground',
'against_ice', 'against_normal', 'against_poison', 'against_psychic',
'against_rock', 'against_steel', 'against_water', 'attack',
'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',    
'classfication', 'defense', 'experience_growth', 'height_m', 'hp',   
'japanese_name', 'name', 'percentage_male', 'pokedex_number',        
'sp_attack', 'sp_defense', 'speed', 'type1', 'type2', 'weight_kg',   
'generation', 'is_legendary']

#TODO:Null study
# print(dataset.isnull().values.any())
# True
# print(dataset.select_dtypes(include='object').columns[dataset.select_dtypes(include='object').isnull().any()])
['type2']
# print(dataset.select_dtypes(include=['int64','float64']).columns[dataset.select_dtypes(include=['int64','float64']).isnull().any()])
['height_m', 'percentage_male', 'weight_kg']

#TODO:Null percentage
# npct=dataset.isnull().sum()/dataset.shape[0]
# print(npct[npct>0])
# height_m           0.024969
# percentage_male    0.122347
# type2              0.479401
# weight_kg          0.024969

#TODO:Dropping irrelevant columns
to_drop=['against_bug', 'against_dark', 'against_dragon',        
'against_electric', 'against_fairy', 'against_fight', 'against_fire',
'against_flying', 'against_ghost', 'against_grass', 'against_ground',
'against_ice', 'against_normal', 'against_poison', 'against_psychic',
'against_rock', 'against_steel', 'against_water','japanese_name','name','pokedex_number']
dataset.drop(to_drop,axis=1,inplace=True)
# print(dataset.shape)
# (801, 20)

#TODO:Filling out null values
dataset['height_m'].fillna(dataset['height_m'].mean(),inplace=True)
dataset['percentage_male'].fillna(dataset['percentage_male'].mean(),inplace=True)
dataset['weight_kg'].fillna(dataset['weight_kg'].mean(),inplace=True)
dataset['type2'].fillna(dataset['type2'].mode()[0],inplace=True)
# dataset.to_csv(r'Datsets\trail.csv')

#TODO:Hot encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype='int')
print(dataset.shape)
