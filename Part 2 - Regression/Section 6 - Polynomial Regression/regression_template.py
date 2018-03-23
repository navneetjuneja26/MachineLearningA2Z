# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Here we have only one independent variable and one dependent variable
# print(X)
# print(y)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
# Building the Linear Regression Model for the dataset

# 3 lines are only used to create the linear regression model
# Poor performance of the linear regression model
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X,y)

# Using the PolynomialFeatures class to perform the Polynomial Regression
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 4)
# # Fit the regressor poly_reg to the X and then transform the object into the X_poly
# X_poly = poly_reg.fit_transform(X)

# lin_reg2 = LinearRegression()
# lin_reg2.fit(X_poly,y)

# print(X_poly)

# Fitting the Regression model to the dataset

# Visualising the linear regression results

# Visualising the Polynomial Regression results

# Predicting a new result with linear regression
# Predicting the salary based on the level
# lin_reg.predict(6.5) # around 330000
# print(lin_reg.predict(6.5))

# Create the regressor


y_pred = regressor.predict(6.5)

# Visualising the regression model results
# (for higher resolution and smoother curves)
X_grid= np.arange(min(X),max(X),0.1)
# To get a matrix use the reshape method
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Regression Model Results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



# Predicting a new result with polynomial regression
# lin_reg2.predict(poly_reg.fit_transform(6.5))
# print(lin_reg2.predict(poly_reg.fit_transform(6.5)))

# Truth and Honest Employee(HR team is happy about the result)




