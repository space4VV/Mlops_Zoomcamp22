#!/usr/bin/env python
# coding: utf-8




import pickle
import pandas as pd
import sys


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def get_predictions(df,year,month):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('mean of the predictions = ',sum(y_pred)/len(y_pred))


    df_final_pq = pd.DataFrame()
    df_final_pq['ride_id'] = df['ride_id']
    df_final_pq['predictions'] = y_pred
    df_final_pq.head()
    df_final_pq.to_parquet('./df_{year}-{month}results.parquet',engine='pyarrow',compression=None,index=False)
    return df_final_pq




if __name__ == '__main__':
    categorical = ['PUlocationID', 'DOlocationID']
    year = sys.argv[1]
    month = sys.argv[2]
    parquet_file = f'./data/fhv_tripdata_{year}-{month}.parquet'
    print(f'calculating the data for the month:{month}and year:{year} using the file:{parquet_file}')
    df =read_data(parquet_file)

    #get predictions and save as df
    df_pred = get_predictions(df,year,month)

    