import pandas as pd
from pandas import DataFrame, Series
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
warnings.simplefilter("always")
simplefilter(action='ignore')

#below code will automatically find the path of this file and accordingly
#form the paths of tic-toc input files since those files
#are also kept at the same location as location of this file
CUR_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
ttt_test = os.path.join(CUR_DIR, 'hw1ttt_test (1).data')
ttt_train = os.path.join(CUR_DIR, 'hw1ttt_train (1).data')
ttt_valid = os.path.join(CUR_DIR, 'hw1ttt_valid (1).data')
ttt_test_csv = os.path.join(CUR_DIR, 'hw1ttt_test (2).csv')
ttt_train_csv = os.path.join(CUR_DIR, 'hw1ttt_train (2).csv')
ttt_valid_csv = os.path.join(CUR_DIR, 'hw1ttt_valid (2).csv')

#function for performing data pre processing on tic toc data set
def data_preprocessing_tic_toc(input_csv, output_csv):
    df= pd.read_csv(input_csv, header=None)
    df.columns=[0,1,2,3,4,5,6,7,8,'train_label']
    df_onehot=pd.get_dummies(df)
    df_onehot.to_csv(output_csv)
    print("data pre-processing for "+input_csv+" file is done.")
    

#function to calculate accuracy & AUROC score using decision tree with gini index
def  decision_tree_gini (csv_file, n):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    #perform training with giniIndex
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=n)
    #Performing training
    clf_gini.fit(X_train, y_train)
    #prediction 
    y_pred = clf_gini.predict(X_test)
    #accuracy calculation
    ac = accuracy_score(y_test,y_pred)
    lr_auc = roc_auc_score(y_test,y_pred)
    print("decision tree accuracy of: "+csv_file+" with gini index for leave node= "+str(n)+" is: "+str(ac)+" and AUROC score is: "+str(lr_auc))

#function to calculate accuracy & AUROC score using decision tree with entropy
def  decision_tree_cross_entropy (csv_file, n):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = n) 
    #Performing training 
    clf_entropy.fit(X_train, y_train) 
    #prediction 
    y_pred = clf_entropy.predict(X_test)
    #accuracy calculation
    ac = accuracy_score(y_test,y_pred)
    lr_auc = roc_auc_score(y_test,y_pred)
    print("decision tree accuracy of: "+csv_file+" with cross entropy for leave node= "+str(n)+" is: "+str(ac)+" and AUROC score is: "+str(lr_auc))

#function to calculate accuracy & AUROC score using SVM
def  SVM_accuracies (csv_file, penalty_value, kernel_value):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
    #train the model
    svclassifier = SVC(C=penalty_value, kernel=kernel_value, gamma='scale')
    svclassifier.fit(X_train, y_train)
    #prediction 
    y_pred = svclassifier.predict(X_test)
    #accuracy calculation
    ac = accuracy_score(y_test,y_pred)
    lr_auc = roc_auc_score(y_test,y_pred)
    print("SVM accuracy of: "+csv_file+" with penalty= "+str(penalty_value)+" & kernel= "+kernel_value+" is: "+str(ac)+" and AUROC score is: "+str(lr_auc))

#function to calculate accuracy & AUROC score using LDA
def  LDA_accuracies (csv_file, penalty_value, kernel_value):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
    #feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #perform LDA
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    #performace comparison with PCA
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #accuracy calculation
    ac = accuracy_score(y_test,y_pred)
    lr_auc = roc_auc_score(y_test,y_pred)
    print("LDA accuracy of: "+csv_file+" is: "+str(ac)+" and AUROC score is: "+str(lr_auc))

#function to perform regression methods such as Best Subset Selection, the LASSO, Ridge Regression, and Principal Components Regression on the Boston dataset
def regression_methods_on_boston():
    #pre-processing
    Boston_data = load_boston()
    Boston = pd.DataFrame(Boston_data.data)
    Boston.columns = Boston_data.feature_names
    Boston = pd.get_dummies(Boston, columns =['CHAS'], drop_first=True)
    X = Boston.drop(columns='CRIM')
    y = Boston['CRIM']
    
    #Lasso regression
    n = 100
    lambdas = (np.logspace(10, -2, num=100))
    lasso = Lasso(normalize = True)
    coefficients = []
    for k in lambdas:
        lassolm = lasso.set_params(alpha = k).fit(X, y)
        coefficients.append(lassolm.coef_)
    np.shape(coefficients)
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    #lambda=0
    lasso0 = Lasso(alpha = 0, normalize = True).fit(X_train, y_train)
    lassopred0 = lasso0.predict(scale(X_test))
    lassocoefs0 = pd.Series(lasso0.coef_, index = X.columns)
    lassointercept0 = pd.Series(lasso0.intercept_, index = ['Intercept'])
    lassotable0 = pd.DataFrame(pd.concat([lassointercept0, lassocoefs0]))
    lassotable0.columns = ['Coefficients']
    lassoerror0 = mean_squared_error(y_test, lassopred0)
    print("mean squared error for LASSO with lambda = 0 is "+str(lassoerror0))
    #selecting lambda through cross validation
    lassocv = LassoCV(alphas = lambdas, normalize = True).fit(X_train, y_train)
    lassocv.alpha_
    # lambda = 0.013219411484660288
    lasso3 = Lasso(alpha = lassocv.alpha_, normalize = True).fit(X_train, y_train)
    lassopred13 = lasso3.predict(scale(X_test))
    lassocoefs13 = pd.Series(lasso3.coef_, index = X.columns)
    lassointercept0013 = pd.Series(lasso3.intercept_, index = ['Intercept'])
    lassotable0013 = pd.DataFrame(pd.concat([lassointercept0013, lassocoefs13]))
    lassotable0013.columns = ['Coefficients']
    lassoerror0001 = mean_squared_error(y_test, lassopred13)
    print("mean squared error for LASSO with lambda = 0.013219411484660288 is "+str(lassoerror0001))


    #Ridge regression
    ridge = Ridge(normalize = True)
    coefs = []
    for k in lambdas:
        ridgelm = ridge.set_params(alpha = k).fit(X, y)
        coefs.append(ridgelm.coef_)
    np.shape(coefs)
    # lambda = 0
    ridge0 = Ridge(alpha = 0, normalize = True).fit(X_train, y_train)
    ridgepred0 = ridge0.predict(scale(X_test))
    ridgecoefs0 = pd.Series(ridge0.coef_, index = X.columns)
    ridgeintercept0 = pd.Series(ridge0.intercept_, index = ['Intercept'])
    ridgetable0 = pd.DataFrame(pd.concat([ridgeintercept0, ridgecoefs0]))
    ridgetable0.columns = ['Coefficients']
    ridgeerror0 = mean_squared_error(y_test, ridgepred0)
    print("mean squared error for ridge regression with lambda = 0 is "+str(ridgeerror0))
    #selecting lambda through cross validation
    ridgecv = RidgeCV(alphas = lambdas, scoring = 'neg_mean_squared_error', normalize = True).fit(X_train, y_train)
    ridgecv.alpha_
    ridge7 = Ridge(alpha = ridgecv.alpha_, normalize = True).fit(X_train, y_train)
    ridgepred7 = ridge7.predict(scale(X_test))
    ridgecoefs7 = pd.Series(ridge7.coef_, index = X.columns)
    ridgeintercept7 = pd.Series(ridge7.intercept_, index = ['Intercept'])
    ridgetable7 = pd.DataFrame(pd.concat([ridgeintercept7, ridgecoefs7]))
    ridgetable7.columns = ['Coefficients']
    ridgeerror002 = mean_squared_error(y_test, ridgepred7)
    print("mean squared error for ridge regression after selecting lambda through cross validation is "+str(ridgeerror002))


    #Principal components regression
    pca = PCA()
    X_scaled = pca.fit_transform(scale(X))
    # selecting the lowest cross-validation error
    n = len(X)
    kf10 = KF(n_splits=10, shuffle=True, random_state=42)
    lm = LinearRegression()
    MSEdf= pd.DataFrame()
    # calculating MSE with only the intercept through cross-validation
    mse = -1*cross_val_score(lm, np.ones((n,1)), y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()    
    MSEdf = MSEdf.append([round(mse, 9)])
    # calculating MSE for the 20 components through cross-validation
    for i in np.arange(1, 21):
        mse = -1*cross_val_score(lm, X_scaled[:,:i], y.ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
        MSEdf = MSEdf.append([round(mse, 9)])
    MSEdf.reset_index(drop=True, inplace=True)
    MSEdf.columns = ['MSE']
    # performing PCR on train and test data sets
    pca_train = PCA()
    X_scaled_train = pca_train.fit_transform(scale(X_train))
    m = len(X_scaled_train)
    lmtrain = LinearRegression()
    kf10train = KF(n_splits=10, shuffle=True, random_state=42)
    MSEdftrain= pd.DataFrame()
    # calculating MSE with only the intercept through cross-validation
    msetrain = -1*cross_val_score(lmtrain, np.ones((m,1)), y_train.ravel(), cv=kf10train, scoring='neg_mean_squared_error').mean()    
    MSEdftrain = MSEdftrain.append([msetrain])
    # calculating MSE for the 20 components through cross-validation
    for i in np.arange(1, 21):
        msetrain = -1*cross_val_score(lmtrain, X_scaled_train[:,:i], y_train.ravel(), cv=kf10train, scoring='neg_mean_squared_error').mean()
        MSEdftrain = MSEdftrain.append([msetrain])
    MSEdftrain.reset_index(drop=True, inplace=True)
    MSEdftrain.columns = ['MSE']
    pca_test = PCA(n_components=12)
    X_scaled_test = pca_test.fit_transform(scale(X_test))
    pcrfit2 = LinearRegression().fit(X_scaled_train, y_train)
    pcrpred2 = pcrfit2.predict(X_scaled_test)
    pcrerror03 = mean_squared_error(y_test, pcrpred2)
    print("mean squared error for PCA after selecting lambda through cross validation is "+str(pcrerror03))


    #comparing MSE of different approaches and selecting one with minimum MSE
    print("Lasso: "+str(lassoerror0001)+" Ridge: "+str(ridgeerror002)+" PCR: "+str(pcrerror03))

   
if __name__ == '__main__':
    
    #data pre-processing of tic toc data set
    data_preprocessing_tic_toc(ttt_test, ttt_test_csv)
    data_preprocessing_tic_toc(ttt_train, ttt_train_csv)
    data_preprocessing_tic_toc(ttt_valid, ttt_valid_csv)

    #calculate decision tree accuracy & AUROC score for tic-toc training data set using gini index
    decision_tree_gini (ttt_train_csv, 1)
    decision_tree_gini (ttt_train_csv, 3)
    decision_tree_gini (ttt_train_csv, 5)
    decision_tree_gini (ttt_train_csv, 15)
    
    #calculate decision tree accuracy & AUROC score for tic-toc validation data set using gini index 
    decision_tree_gini (ttt_valid_csv, 1)
    decision_tree_gini (ttt_valid_csv, 3)
    decision_tree_gini (ttt_valid_csv, 5)
    decision_tree_gini (ttt_valid_csv, 15)

    #calculate decision tree accuracy & AUROC score for tic-toc testing data set using gini index
    decision_tree_gini (ttt_test_csv, 1)
    decision_tree_gini (ttt_test_csv, 3)
    decision_tree_gini (ttt_test_csv, 5)
    decision_tree_gini (ttt_test_csv, 15)

    #calculate decision tree accuracy & AUROC score for tic-toc training data set using cross entropy
    decision_tree_cross_entropy (ttt_train_csv, 1)
    decision_tree_cross_entropy (ttt_train_csv, 3)
    decision_tree_cross_entropy (ttt_train_csv, 5)
    decision_tree_cross_entropy (ttt_train_csv, 15)
    

    #calculate decision tree accuracy & AUROC score for tic-toc validation data set using cross entropy 
    decision_tree_cross_entropy (ttt_valid_csv, 1)
    decision_tree_cross_entropy (ttt_valid_csv, 3)
    decision_tree_cross_entropy (ttt_valid_csv, 5)
    decision_tree_cross_entropy (ttt_valid_csv, 15)

    #calculate decision tree accuracy & AUROC score for tic-toc testing data set using cross entropy
    decision_tree_cross_entropy (ttt_test_csv, 1)
    decision_tree_cross_entropy (ttt_test_csv, 3)
    decision_tree_cross_entropy (ttt_test_csv, 5)
    decision_tree_cross_entropy (ttt_test_csv, 15)

    #calculate SVM accuracy & AUROC score for tic-toc training data set 
    SVM_accuracies (ttt_train_csv, 0.1, 'poly')
    SVM_accuracies (ttt_train_csv, 0.2, 'poly')
    SVM_accuracies (ttt_train_csv, 0.1, 'rbf')
    SVM_accuracies (ttt_train_csv, 0.2, 'rbf')
    SVM_accuracies (ttt_train_csv, 0.1, 'sigmoid')
    SVM_accuracies (ttt_train_csv, 0.2, 'sigmoid')
    
    #calculate SVM accuracy & AUROC score for tic-toc validation data set  
    SVM_accuracies (ttt_valid_csv, 0.1, 'poly')
    SVM_accuracies (ttt_valid_csv, 0.2, 'poly')
    SVM_accuracies (ttt_valid_csv, 0.1, 'rbf')
    SVM_accuracies (ttt_valid_csv, 0.2, 'rbf')
    SVM_accuracies (ttt_valid_csv, 0.1, 'sigmoid')
    SVM_accuracies (ttt_valid_csv, 0.2, 'sigmoid')

    #calculate SVM accuracy & AUROC score for tic-toc testing data set 
    SVM_accuracies (ttt_test_csv, 0.1, 'poly')
    SVM_accuracies (ttt_test_csv, 0.2, 'poly')
    SVM_accuracies (ttt_test_csv, 0.1, 'rbf')
    SVM_accuracies (ttt_test_csv, 0.2, 'rbf')
    SVM_accuracies (ttt_test_csv, 0.1, 'sigmoid')
    SVM_accuracies (ttt_test_csv, 0.2, 'sigmoid')

    #calculate LDA accuracy & AUROC score for tic-toc training data set 
    LDA_accuracies (ttt_train_csv, 0.1, 'poly')
    
    #calculate LDA accuracy & AUROC score for tic-toc validation data set  
    LDA_accuracies (ttt_valid_csv, 0.1, 'poly')
   

    #calculate LDA accuracy & AUROC score for tic-toc testing data set 
    LDA_accuracies (ttt_test_csv, 0.1, 'poly')

    #Regression methods on Boston data set
    regression_methods_on_boston()
  
    



    

    



