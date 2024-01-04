#!/usr/bin/env python3
"""
preprocess this data
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def preprocess_data(file):
    # Load dataset
    df = pd.read_csv(file)

    # Print column names to inspect
    print(df.columns)

    # Modify column names based on your dataset
    df = df[['Open', 'High', 'Low', 'Close',
             'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]

    # Normalize data using Min-Max scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Add target column (next hour's closing price)
    df_scaled['target'] = df_scaled['Close'].shift(-1)

    # Drop the last row with NaN target
    df_scaled = df_scaled[:-1]

    # Split the dataset into training and testing sets
    X = df_scaled.drop('target', axis=1)
    y = df_scaled['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    # Save preprocessed data
    np.savez('preprocessed_data.npz', X_train=X_train.values,
             y_train=y_train.values, X_test=X_test.values, y_test=y_test.values)


if __name__ == "__main__":
    preprocess_data('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
