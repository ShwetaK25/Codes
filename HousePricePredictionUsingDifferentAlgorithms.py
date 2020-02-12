import pandas as pd
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier
from sklearn. metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import ensemble
import matplotlib
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2, mean_squared_error
from sklearn.preprocessing import scale
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold as KF, cross_val_score
from warnings import simplefilter
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_boston
from sklearn import metrics
  
#below code will automatically find the path of this file and accordingly
#form the paths of other input files assuming that those files
#are also kept at the same location as location of this file
CUR_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
input_csv = os.path.join(CUR_DIR, 'washington1.csv')
output_csv = os.path.join(CUR_DIR, 'washington1_output_csv.csv')
input_csv2 = os.path.join(CUR_DIR, 'boston_housing.csv')
output_csv2 = os.path.join(CUR_DIR, 'boston_housing_csv.csv')
input_csv3 = os.path.join(CUR_DIR, 'washington2.csv')
output_csv3 = os.path.join(CUR_DIR, 'washington2_csv.csv')


#function for performing data pre processing on washington data set
def data_preprocessing(input_csv, output_csv):
    dataframe = pd.read_csv(input_csv)
    del dataframe['country']
    dataframe['date'] = dataframe['date'].str.replace('00:00','')
    dataframe['statezip'] = dataframe['statezip'].str.replace('WA','')
    df_onehot=pd.get_dummies(dataframe)
    df_onehot.to_csv(output_csv)
    print("data pre-processing for "+input_csv+" file is done.")

#function for performing data pre processing on boston data set
def data_preprocessing2(input_csv, output_csv):
    dataframe = pd.read_csv(input_csv)
    df_onehot = pd.get_dummies(dataframe)
    df_onehot.to_csv(output_csv)
    print("data pre-processing for "+input_csv+" file is done.")

#Function to perform model implementations on Boston data set
def boston_exrcise(output_csv2):
    data = pd.read_csv(output_csv2)
    prices = data['MEDV']
    features = data.drop('MEDV', axis = 1)

    #Data splitting
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)

    #Decision tree regression 
    regressor = DecisionTreeRegressor()
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    grid = GridSearchCV(estimator=regressor, cv=cv_sets, param_grid = params)
    grid = grid.fit(X_train, y_train)
    ac = grid.score(X_test, y_test)
    print("Decision tree-regression accuracy of: "+output_csv2+" is: "+str(ac))

    #Linear Regression
    reg2= LinearRegression()
    reg2.fit(X_train, y_train)
    y_pred = reg2.predict(X_test)
    ac2 = metrics.mean_squared_error(y_test, y_pred)
    print("Linear regression mean sqared error of: "+output_csv2+" is: "+str(ac2))

    #Gradient boosting algorithm
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth = 5, min_samples_split =2, loss='ls', learning_rate=0.1)
    clf.fit(X_train, y_train)
    ac3 = clf.score(X_test, y_test)
    print("Gradient boosting algorithm - accuracy of: "+output_csv2+" is: "+str(ac3))

#Function to perform model implementations on Washington data set
def WashingtonModels(csv_file):
    data = pd.read_csv(csv_file)
    prices = data['price']
    features = data.drop('price', axis = 1)

    #Data spliting
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state = 42)

    #Decision tree regression 
    regressor = DecisionTreeRegressor()
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    grid = GridSearchCV(estimator=regressor, cv=cv_sets, param_grid = params)
    grid = grid.fit(X_train, y_train)
    ac = grid.score(X_test, y_test)
    print("Decision tree-regression accuracy of: "+csv_file+" is: "+str(ac))
    
    #Linear Regression
    reg2= LinearRegression()
    reg2.fit(X_train, y_train)
    y_pred = reg2.predict(X_test)
    ac2 = metrics.mean_squared_error(y_test, y_pred)
    print("Linear regression mean sqared error of: "+csv_file+" is: "+str(ac2))

    #Gradient boosting algorithm
    clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth = 5, min_samples_split =2, loss='ls', learning_rate=0.1)
    clf.fit(X_train, y_train)
    ac3 = clf.score(X_test, y_test)
    print("Gradient boosting algorithm - accuracy of: "+csv_file+" is: "+str(ac3))
   


if __name__ == '__main__':
      
    #Data pre-processing 
    data_preprocessing(input_csv, output_csv)
    data_preprocessing2(input_csv2, output_csv2)
    data_preprocessing2(input_csv3, output_csv3)
    
    #Calculate accuracy scores
    WashingtonModels(output_csv)
    boston_exrcise(output_csv2)
    WashingtonModels(output_csv3)

    

   
    



