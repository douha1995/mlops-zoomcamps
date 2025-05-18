import pandas as pd
import sklearn
# print(pd.__version__)

df_trips = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet')

print(df_trips.head())

print(sklearn.__version__)