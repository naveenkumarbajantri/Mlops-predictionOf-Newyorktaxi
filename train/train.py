#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import mlflow
from mlflow.exceptions import MlflowException


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    print(f'number of rows for {filename} is {len(df)}')

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def train(train_date='2022-01', validation_date='2022-02'):
    train_path = f'./green_tripdata_{train_date}.parquet'
    validation_path = f'./green_tripdata_{validation_date}.parquet'

    df_train = read_dataframe(train_path)
    df_val = read_dataframe(validation_path)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    with mlflow.start_run():
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']

        train_dicts = df_train[categorical + numerical].to_dict(orient='records')
        val_dicts = df_val[categorical + numerical].to_dict(orient='records')

        mlflow.log_params({
            'categorical': categorical,
            'numerical': numerical,
        })

        pipeline = make_pipeline(
            DictVectorizer(),
            LinearRegression()
        )

        pipeline.fit(train_dicts, y_train)
        y_pred = pipeline.predict(val_dicts)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'✅ RMSE on validation is {rmse}')
        mlflow.log_metric('rmse', rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path='model')  # Save model


def run():
    # ✅ Use relative mlruns path compatible with GitHub Actions
    tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
    mlflow.set_tracking_uri(tracking_path)

    experiment_name = "nyc-taxi-experiment"

    try:
        mlflow.create_experiment(name=experiment_name)
    except MlflowException:
        pass  # already exists

    mlflow.set_experiment(experiment_name)
    train(train_date='2022-01', validation_date='2022-02')


if __name__ == '__main__':
    run()
