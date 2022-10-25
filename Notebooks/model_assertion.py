import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#GetStrategy
import xgboost as xgb

#ValidateStrategy
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan

from yfinance_data_func import *
from trading_strategy_sgb import *
from trading_strategy_validation import *

class ModelAssertion:
    '''
    Use different machine learning models to detect a huge surge in error metrics
    window_out_train: length of the period to train different models
    window_out_test: length of the period to test different models
    '''
    def __init__(self,df_result,window_me):
        '''
        df_result: DataFrame contains ['Date','MAE','MSE', 'MAPE', 'sMAPE']
        window_me: length of the period to calculate error metrics
        '''
        self.df_result = df_result 
        self.df_Valid = ValidateStrategy.get_df(window_me,df_result)
   
    def outlier_mean_std(self,window_out_train,window_out_test, std_num):
        '''
        model assertion warning: 
        the mean of all error metrics in test are larger than the mean + 3* std of all error metrcs in train
        '''
        X = self.df_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]]
        X_train = X[:window_out_train]
        thres = std_num*X_train.std()+X_train.mean()
        for index in range(window_out_train+window_out_test,len(self.df_Valid)-1):
            X_test = X[index-window_out_test:index]
            X_test_mean = X_test.mean()
            test_result = X_test_mean>thres
            if (test_result == True).all() == True:
                print(self.df_Valid.loc[index, 'Date'])
                return self.df_Valid.loc[index, 'Date']
    def outlier_iso_forest(self,window_out_train,window_out_test):
        '''
        model assertion warning: 
        1. Normalize train and test dataset
        2. Train isolation forest in train datasets. 
        3. If all datapoints in test datasets are detected as outliers by isolation forest moodels, break!
        '''
        X = self.df_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]]
        X_train = X[:window_out_train]
        #use train datasets to train StandardScaler
        scaler_1 = StandardScaler()
        scaler_1.fit(np.array(X_train['MAE']).reshape(-1,1))
        scaler_2 = StandardScaler()
        scaler_2.fit(np.array(X_train['MSE']).reshape(-1,1))
        scaler_3 = StandardScaler()
        scaler_3.fit(np.array(X_train['MAPE']).reshape(-1,1))
        scaler_4 = StandardScaler()
        scaler_4.fit(np.array(X_train['sMAPE']).reshape(-1,1))
        X_train['MAE'] = X_train['MAE'].apply(lambda x: scaler_1.transform(np.array(x).reshape(-1,1))[0][0])
        X_train['MSE'] = X_train['MSE'].apply(lambda x: scaler_2.transform(np.array(x).reshape(-1,1))[0][0])
        X_train['MAPE'] = X_train['MAPE'].apply(lambda x: scaler_3.transform(np.array(x).reshape(-1,1))[0][0])
        X_train['sMAPE'] = X_train['sMAPE'].apply(lambda x: scaler_4.transform(np.array(x).reshape(-1,1))[0][0])
        clf = IsolationForest(random_state=0).fit(X_train)
        for index in range(window_out_train+window_out_test,len(self.df_Valid)-1):
            X_test = X[index-window_out_test:index]
            #apply trained StandardScaler to test datasets
            X_test['MAE'] = X_test['MAE'].apply(lambda x: scaler_1.transform(np.array(x).reshape(-1,1))[0][0])
            X_test['MSE'] = X_test['MSE'].apply(lambda x: scaler_2.transform(np.array(x).reshape(-1,1))[0][0])
            X_test['MAPE'] = X_test['MAPE'].apply(lambda x: scaler_3.transform(np.array(x).reshape(-1,1))[0][0])
            X_test['sMAPE'] = X_test['sMAPE'].apply(lambda x: scaler_4.transform(np.array(x).reshape(-1,1))[0][0])
            test_result = clf.predict(X_test).tolist()
            #check if all datapoints in test datasets are detected as outliers
            if all(i == -1 for i in test_result) == True:
                print(self.df_Valid.loc[index, 'Date'])
                return self.df_Valid.loc[index, 'Date']

    def outlier_dbscan(self, window_out_train,window_out_test, DF_SPY_result_Valid):
        '''
        model assertion warning: 
        1. Combine train and test dataseat together 
        2. Normalize combined dataset and use dbscan to assign combined data into different groups
        3. If groups in train and test datasets are completely different, break!
        '''
        X = self.df_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]]
        for index in range(window_out_train+window_out_test,len(self.df_Valid)-1):
            #combine train and test data
            X = pd.concat([DF_SPY_result_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]][:window_out_train],\
                           DF_SPY_result_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]][index-window_out_test:index]])
            #Normalize
            for i in ['MAE','MSE', 'MAPE', 'sMAPE' ]:
                scaler = StandardScaler()
                scaler.fit(np.array(X[i]).reshape(-1,1))
                X[i] = X[i].apply(lambda x: scaler.transform(np.array(x).reshape(-1,1))[0][0])
            #Train model
            dbscan=DBSCAN()
            model = dbscan.fit(X)
            #get unique groups of train and test data
            train_result = model.labels_[:window_out_train].tolist()
            test_result = model.labels_[-window_out_test:].tolist()
            train_unique = list(set(train_result))
            train_unique.remove(-1) if -1 in train_unique else None
            test_unique = list(set(test_result))
            test_unique.remove(-1) if -1 in test_unique else None
            #print(model.labels_)
            #print(train_unique,test_unique)
            #check if groups in train and test datasets are completely different
            if any(x in test_unique for x in train_unique)==False:
                print(self.df_Valid.loc[index, 'Date'])
                return self.df_Valid.loc[index, 'Date']

    def outlier_hdbscan(self, window_out_train,window_out_test, DF_SPY_result_Valid):
        '''
        model assertion warning: 
        1. Combine train and test dataseat together 
        2. Normalize combined dataset and use dbscan to assign combined data into different groups
        3. If groups in train and test datasets are completely different, break!
        '''
        X = self.df_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]]
        for index in range(window_out_train+window_out_test,len(self.df_Valid)-1):
            #combine train and test data
            X = pd.concat([DF_SPY_result_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]][:window_out_train],\
                           DF_SPY_result_Valid[['MAE','MSE', 'MAPE', 'sMAPE' ]][index-window_out_test:index]])
            #Normalize
            for i in ['MAE','MSE', 'MAPE', 'sMAPE' ]:
                scaler = StandardScaler()
                scaler.fit(np.array(X[i]).reshape(-1,1))
                X[i] = X[i].apply(lambda x: scaler.transform(np.array(x).reshape(-1,1))[0][0])
            #Train model
            clusterer = hdbscan.HDBSCAN()
            model = clusterer.fit(X)
            #get unique groups of train and test data
            train_result = model.labels_[:window_out_train].tolist()
            test_result = model.labels_[-window_out_test:].tolist()
            train_unique = list(set(train_result))
            train_unique.remove(-1) if -1 in train_unique else None
            test_unique = list(set(test_result))
            test_unique.remove(-1) if -1 in test_unique else None
            #print(model.labels_)
            #print(train_unique,test_unique)
            if any(x in test_unique for x in train_unique)==False:
                print(self.df_Valid.loc[index, 'Date'])
                return self.df_Valid.loc[index, 'Date']