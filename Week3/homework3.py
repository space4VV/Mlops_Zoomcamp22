import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime
import os
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta


@task
def read_data(path):
    df = pd.read_parquet(path, engine="pyarrow")
    return df


@task
def get_paths(date):
    logger = get_run_logger()
    mypath = "./data"
    filenames = next(os.walk(mypath))[2]
    # date = datetime.now().strftime('%Y-%m-%d')
    date_format = "%Y-%m-%d"
    train_cond = relativedelta(months=2)
    val_cond = relativedelta(months=1)
    for file in filenames:

        if date == None:
            logger.info("No date given , setting todays date")

            date = datetime.now().strftime("%Y-%m-%d")

        dateObj = datetime.strptime(date, date_format)
        train_month = (dateObj - train_cond).strftime("%Y-%m")
        val_month = (dateObj - val_cond).strftime("%Y-%m")

        match = re.search(r"\d{4}-\d{2}", file)
        if train_month == str(match.group()):
            train_path = str(mypath + "/" + file)
        if val_month == str(match.group()):
            val_path = str(mypath + "/" + file)
    logger.info("train_path", train_path)
    logger.info("val_path", val_path)
    return train_path, val_path


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())

def main(date="2021-08-15"):

    categorical = ["PUlocationID", "DOlocationID"]

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)

    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)


main()
