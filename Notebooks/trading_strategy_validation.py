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

class ValidateStrategy:
    '''
    Given y_pred and y_test, calculate ['MAE','MSE', 'MAPE', 'sMAPE' ] for each single day in a rolling base
    '''
    def metric(data):
        '''
        data: DataFrame containing Date, y_test and y_pred
        '''
        test = data.y_test
        pred = data.y_pred
        d = [('MAE', mean_absolute_error(test, pred)),
             ('MSE', mean_squared_error(test, pred)),
             ('MAPE', mean_absolute_percentage_error(test, pred, symmetric=False)),
             ('sMAPE', mean_absolute_percentage_error(test, pred, symmetric=True))]
        data_new = pd.DataFrame(d, columns=['Metrics', 'Value']).set_index('Metrics')
        return data_new
    def get_metric(w,i,df):  
        '''
        Calculate error metrics for each single day
        w: rolling window for calculating error metrics. In other words, w days before i.
        i: the index of the day we want to calculate error metrics
        df: DataFrame containing Date, y_test and y_pred
        (If i is smaller than w, just use the days it has before i.)
        '''
        if i >= w:
            start = i-w+1
            end = i+1
            return ValidateStrategy.metric(df.iloc[start:end,1:])
        else: 
            return ValidateStrategy.metric(df.iloc[:i+1,1:])
    def get_df(window,df_result):
        '''
        Combine error metrics of all days into one DataFrame
        window: rolling window for calculating error metrics.
        df_result:DataFrame containing Date, y_test and y_pred
        '''
        df_result = df_result.reset_index()
        for index in df_result.index:
            df_result.loc[index,'MAE'] = ValidateStrategy.get_metric(w=window, i=index, df=df_result).loc['MAE','Value']
            df_result.loc[index,'MSE'] = ValidateStrategy.get_metric(w=window, i=index, df=df_result).loc['MSE','Value']
            df_result.loc[index,'MAPE'] = ValidateStrategy.get_metric(w=window, i=index, df=df_result).loc['MAPE','Value']
            df_result.loc[index,'sMAPE'] = ValidateStrategy.get_metric(w=window, i=index, df=df_result).loc['sMAPE','Value']
        return df_result