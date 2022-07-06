import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from train_model import * 

def performance_on_test(model,X_test,y_test):
  mean_sq_err = 0; mean_abs_err = 0;

  model.fit(X_test, y_test)
  y_test_predict = model.predict(X_test)
  mean_sq_err = mean_squared_error(y_test,y_test_predict,squared=True)
  mean_abs_err = mean_absolute_error(y_test,y_test_predict)

  return mean_sq_err, mean_abs_err

categorical_col, numerical_col = categorical_and_numerical_columns_regression(car_price_dataset)
X_train, y_train, X_test, y_test = prepare_data_regression(train, test, categorical_col, numerical_col)
lin_reg_model = fit_model_to_the_data(X_train,y_train)

mean_sq_err, mean_abs_err = performance_on_test(lin_reg_model, X_test,y_test)
print("Mean squared error: %.2f" % mean_sq_err)
print("Mean absolute error: %.2f" % mean_abs_err)