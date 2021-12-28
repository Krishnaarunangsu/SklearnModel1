# Reference: https://towardsdatascience.com/simple-linear-regression-35b3d940950e
# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Fetch the data
def fetch_data():
    """Fetch the Data"""
    data = pd.read_csv("..//simple_linear_regression//data//advertising.csv")
    # print(data.head())
    # print(data['TV'].head())
    return data


# Defining X and Y
fetched_data = fetch_data()
X = fetched_data['TV'].values.reshape(-1, 1)
# print(X)
Y = fetched_data['Sales'].values.reshape(-1, 1)

# Splitting the data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Fitting the model on Train Dataset
Model = LinearRegression(fit_intercept=True)
Model = Model.fit(X_train, y_train)

print("Intercept is ", Model.intercept_[0])
print("Coefficient is ", Model.coef_[0][0])

# Predicting and storing results for Test Dataset
train_fit = Model.predict(X_train)
test_pred = Model.predict(X_test)
# print(train_fit)
# print("***************************************************")
# print(test_pred)

plt.figure(figsize=(12, 4))
# Plotting Regression line on Train Dataset
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='gray')
plt.plot(X_train, train_fit, color='blue', linewidth=2)
plt.xlabel('TV')
plt.ylabel('sales')
plt.title("Train Dataset")
# Plotting Regression line on Test Dataset
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, test_pred, color='blue', linewidth=2)
plt.xlabel('TV')
plt.ylabel('sales')
plt.title("Test Dataset")
plt.show()
