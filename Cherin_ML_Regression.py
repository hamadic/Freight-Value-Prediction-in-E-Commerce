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

#import performance metric functions
from sklearn.metrics import confusion_matrix as cm


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

df1= df1.sample(n=5000,replace="False")##
# check correlation 
plt.matshow(df1.corr())

import seaborn as sns
corr = df1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

pd.scatter_matrix(df1, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# define independent variable and dependent variable
X = df1.iloc[:, 1: ].values
y = df1.iloc[:, 0].values

#feature scaling for independent variables
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state=0)

#Fitting linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
Linear=lr.fit(X_train,y_train)
print (lr.intercept_)
print (lr.coef_)
print (lr.score(X_train, y_train))

#Predicting on test set
y_pred = lr.predict(X_test)
print(lr.score(X_test,y_test))

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
accuricies = cross_val_score(estimator = lr, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies_mean = accuricies.mean()
accuricies.std()

# Gridsearch for hyperparameter tuning using Ridge regularization for linear regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

model= Ridge()
alphas = np.array([1,0.15, 0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X_train, y_train)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha) 

###############################################################################

# Polynomial Regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)


from math import sqrt
from sklearn.preprocessing import PolynomialFeatures


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#Transformation and Regressioin with Degree-3

poly = PolynomialFeatures(degree = 3)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred1 = lin_reg.predict(X_test_poly)
lin_reg.score(X_train_poly, y_train)

lin_reg.score(X_test_poly, y_test)

#### polynomial gridsearch 

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm, grid_search

import numpy as np 
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline 

def PolynomialRegression(degree=2, **kwargs): 
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs)) 

param_grid = {'polynomialfeatures__degree': np.arange(5), 'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]} 

Pl=PolynomialRegression()
poly_grid = GridSearchCV(Pl, param_grid) 

poly_grid.fit(X_train_poly, y_train)

print(poly_grid)
print(poly_grid.best_score_)
poly_grid.best_params_
poly_grid.best_estimator_

#Cross Validation using K-fold 
from sklearn.model_selection import KFold

kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_poly, X_test_poly = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator = Pl, X= X_train_poly, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies_mean = accuricies.mean()
accuricies.std()

##########################################################################################################



