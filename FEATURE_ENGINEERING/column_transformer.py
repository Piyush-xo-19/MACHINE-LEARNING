import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"C:\Users\LENOVO\Desktop\machine learnig\FEATURE_ENGINEERING\covid_toy.csv")
from sklearn.model_selection import train_test_split
X=df.iloc[0::,5]
y= df["has_covid"]
X_train , X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=42)
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])
],remainder='passthrough')
transformer.fit_transform(X_train).shape()
transformer.fit_transform(X_test).shape()
print(X_test.shape())
print(X_train.shape())