import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values #independant vars
y = dataset.iloc[:, -1].values #dependant vars
print('x', x)
print('y', y)

# missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])
print('updated x', x)

#encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print('encoded x', x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print('encoded y', y)

# training set vs test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print('x_train', x_train)
print('x_test', x_test)
print('y_train', y_train)
print('y_train', y_test)

#feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train[:, 3:] = ss.fit_transform(x_train[:, 3:])
x_test[:, 3:] = ss.fit_transform(x_test[:, 3:])

print('x_train', x_train)
print('x_test', x_test)