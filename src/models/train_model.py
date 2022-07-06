# -*- coding: utf-8 -*-
"""DA212_PS2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16D7iG9WeNjZ3VHHzQbRgVJ0G_pgrUq9I
"""

#packages imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

car_price_dataset = pd.read_csv("https://www.dropbox.com/s/0vvhzsbf0tmjq2s/CarPrice_Dataset_cleaned.csv?dl=1")

train, test = train_test_split(car_price_dataset, test_size=0.25, random_state=100)

def categorical_and_numerical_columns_regression(car_price_dataset):
  categorical_col = []
  numerical_col = []

  for i in car_price_dataset.describe(include = 'object').columns:
    categorical_col.append(i)

  for i in car_price_dataset.describe().columns:
    if(i != 'price'):
      numerical_col.append(i)


  return categorical_col, numerical_col

def prepare_data_regression(train, test, categorical_col, numerical_col):

  X_train = np.array([]); y_train = np.array([]); X_test = np.array([]); y_test = np.array([]);

  X_train = train.copy().drop("price", axis=1)
  y_train = train["price"].copy()
  X_test = test.copy().drop("price", axis=1)
  y_test = test["price"].copy()

  trainDf_Catg = train[categorical_col].copy()
  testDf_Catg = test[categorical_col].copy()
  trainDf_Num = train[numerical_col].copy()
  testDf_Num = test[numerical_col].copy()

  #feature scaling of numerical features
  scaler = StandardScaler()
  trainDf_Num_scl = scaler.fit_transform(trainDf_Num)
  testDf_Num_scl = scaler.transform(testDf_Num)

  #one-hot encoding of categorical features.
  ohe = OneHotEncoder()
  trainDf_Catg_1hot = ohe.fit_transform(trainDf_Catg)
  testDf_Catg_1hot = ohe.transform(testDf_Catg)

  #Join back scaled and Encoded Numerical and categorical data
  X_train = np.concatenate([trainDf_Num_scl,trainDf_Catg_1hot.toarray()],axis=1)
  X_test = np.concatenate([testDf_Num_scl,testDf_Catg_1hot.toarray()],axis=1)

  return X_train, y_train, X_test, y_test

def fit_model_to_the_data(X,y):
  model = LinearRegression()
  model.fit(X, y)

  return model

def lin_reg_parameters(model):
  intercept = 0; coef = 0

  intercept = model.intercept_
  coef = model.coef_

  return intercept, coef

categorical_col, numerical_col = categorical_and_numerical_columns_regression(car_price_dataset)
X_train, y_train, X_test, y_test = prepare_data_regression(train, test, categorical_col, numerical_col)
lin_reg_model = fit_model_to_the_data(X_train,y_train)