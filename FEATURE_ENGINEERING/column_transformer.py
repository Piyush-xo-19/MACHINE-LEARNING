import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df=pd.read_csv(r"C:\Users\LENOVO\Desktop\machine learnig\FEATURE_ENGINEERING\covid_toy.csv")
from sklearn.model_selection import train_test_split
X=df.iloc[0::,5]
y= df["has_covid"]
X_train , X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=42)
