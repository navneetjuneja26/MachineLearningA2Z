# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
# specify the working directory folder
dataset = pd.read_csv('Data.csv')
# : implies take all the values
# :-1 represent take all the values except the last one
# This is the matrix of the features
X = dataset.iloc[:,:-1].values
# Creating the dependable variable vector
Y = dataset.iloc[:,3].values
# Replace the missing data with the mean of the values in that column
from sklearn.preprocessing import Imputer
#Create an object of the class
#Putting missing values as NaN
#Chosing axis as 0 as we want the mean of the columns
#imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
#Fit the imputer where there is some missing data
#imputer = imputer.fit(X[:,1:3])
#Replace the missing data of the matrix X with the mean of the column
#X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
#here it is country and the purchased field
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Create the object of the LabelEncoder
#labelencoder_X = LabelEncoder()
#Encode the coulmns using the fit transform
##X[:,0] = labelencoder_X.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
#Dummy encode the purchased column
#create the label encoder object
#labelencoder_y = LabelEncoder()
#Y = labelencoder_y.fit_transform(Y)

#Splitting the dataset into the training set and the test set
#Import the library that will split the dataset for us
from sklearn.cross_validation import train_test_split
#test_size determines how much data goes to the training set and the testing set
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size= 0.2,random_state=0)

#Feature Scaling
""""from sklearn.preprocessing import StandardScaler
sc_X =  StandardScaler()
# Rescale the X_train 
#For training set we need to fit it and then transform
#For test set we only transform
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""






























