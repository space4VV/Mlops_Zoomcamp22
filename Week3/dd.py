import pandas as pd

df = pd.read_parquet("data/fhv_tripdata_2021-06.parquet", engine="pyarrow")
print(df.head())
