from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataSet:
    def __init__(self, data):
        self.columns = ['Date', 'open_val', 'high_val', 'low_val', 'volume_val', 'close_val']
        self.train = None
        self.test = None
        self.window_size = 100
        self.scaler = MinMaxScaler((-10, 10))
        df = pd.DataFrame(columns=self.columns,
                          data=data)
        df.set_index('Date', inplace=True)
        self.scaler.fit(df[['open_val', 'high_val', 'low_val', 'volume_val', 'close_val']])

    def preprocess(self, data, test_ratio=0.2):
        df = pd.DataFrame(columns=self.columns,
                               data=data)
        df.set_index('Date', inplace=True)
        scaled_data = self.scaler.transform(
            df[['open_val', 'high_val', 'low_val', 'volume_val', 'close_val']])

        train_size = int(np.ceil(df.shape[0] * (1 - test_ratio)))
        self.train = scaled_data[:train_size]
        self.test = scaled_data[train_size:]

        X_train, y_train = self.split(self.train, self.window_size, self.window_size)
        X_test, y_test = self.split(self.test, self.window_size, self.window_size)
        return X_train, y_train, X_test, y_test

    def split(self, data, N, offset):
        X, y = [], []

        for i in range(offset, len(data)):
            X.append(data[i - N:i])
            y.append(data[i])
        return np.array(X), np.array(y)


