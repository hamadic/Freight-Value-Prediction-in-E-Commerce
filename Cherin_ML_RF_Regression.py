# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:17:43 2018

@author: Cherin
"""

import os
os.getcwd()
os.chdir(r'C:\Users\Cherin\Desktop\Metro College\MachineLearning')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#libraries for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Import Dataframe
df= pd.read_csv ('Ecommerce.csv', na_values='')
df.isnull().sum().sum()

# Preprocessing

# slicing dataset
df.columns
df1=df.loc[:,["order_freight_value","order_products_value", "order_items_qty","product_name_lenght", "product_description_lenght",
             "product_photos_qty", "review_score"]]

#Check if there is missing data
df1.isnull().sum().sum()
df1= df1.sample(n=5000,replace="False")

# check correlation 
plt.matshow(df1.corr())

import seaborn as sns
corr = df1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# define independent variable and dependent variable
X = df1.iloc[:, 1: ].values
y = df1.iloc[:, 0].values

#feature scaling for independent variables
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state=0)

#Fitting Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
# Predicting the Test set results
y_pred = rf.predict(X_test)
rf.score(X_test,y_test)


#Cross Validation using K-fold 
from sklearn.model_selection import KFold

kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator = rf, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies_mean = accuricies.mean()
accuricies.std()

# run gridsearch
from sklearn.model_selection import GridSearchCV
params = {
            'n_estimators':[1, 10, 100],
            'max_depth':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
            
        }
rf1=RandomForestRegressor()
rf = GridSearchCV(rf1,params)
rf.fit(X_train,y_train)
rf.best_params_
rf.best_score_


