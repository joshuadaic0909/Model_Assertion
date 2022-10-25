class get_yfinance_data:
    def __init__(self, ticker_symbol, start_date, end_date):
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        # get daily financial data from yahoo finance api
        self.df = yf.download(self.ticker_symbol, 
                      start=self.start_date, 
                      end=self.end_date, 
                      progress=False)
    def add_techical_indicators(self):
        new_df = self.df
        
        # date
        new_df['year'] = new_df.index.get_level_values('Date').year
        new_df['month'] = new_df.index.get_level_values('Date').month
        
        #moving average
        new_df['MA'] = MA(new_df.Close, timeperiod=30, matype=0)
        
        # MIDPRICE - Midpoint Price over period
        new_df['MIDPRICE'] = MIDPRICE(new_df.High, new_df.Low, timeperiod=14)

        
        # Rolling Average Dollar Volume
        new_df['dollar_vol'] = new_df[['Close', 'Volume']].prod(axis=1)
        new_df['dollar_vol_1m'] = (new_df.dollar_vol
                                .rolling(window=30)
                                .mean()).values
        
        # Bollinger Bands
        new_df['upperband'],new_df['middleband'],new_df['lowerband'] = BBANDS(new_df.Close, 
                                                                     timeperiod=5, 
                                                                     nbdevup=2, 
                                                                     nbdevdn=2, 
                                                                     matype=0)
        # Momentum Indicators
        new_df['RSI'] = RSI(new_df.Close, timeperiod=14)
        new_df['MACD'] = MACD(new_df.Close, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        new_df['ATR'] = ATR(new_df.High, new_df.Low, new_df.Close, timeperiod=14)
        
        new_df.dropna(inplace=True)
        
        return new_df