import xgboost as xgbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

#FeatureEngineering
#from talib import RSI, BBANDS, MACD, ATR, MA, MIDPRICE

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

class Strategy_generation:
    def __init__(self, df_stock, split_date):
        import xgboost as xgb
        self.split_date = split_date
        self.df_stock = df_stock
        # we shift -1 because we want to predict the price of the next day
        self.df_stock['y'] = self.df_stock['Close'].shift(-1)
        X = self.df_stock.dropna()
        y = X['y']
        X = X.drop('y', axis = 1)
        # Train test split
        self.X_train, self.X_test = X[X.index < self.split_date], X[X.index >= self.split_date]
        self.y_train, self.y_test = y[y.index < self.split_date], y[y.index >= self.split_date]

    def train_XgbRegressor(self):
        '''
        We train on all the technical indicators we created from the feature engineering steps
        '''
        base = xgbf.XGBRegressor()
        model = base.fit(self.X_train, self.y_train.values.ravel())
        self.model = model
        return model

    def run_XgbRegressor(self):
        ypred = self.model.predict(self.X_test)
        ypred_s = pd.Series(ypred, self.y_test.index)
        result = pd.concat([ypred_s, self.y_test], axis=1)
        result.columns = ['y_pred','y_test']
        self.ypred_s = ypred_s
        self.result = result
        return result