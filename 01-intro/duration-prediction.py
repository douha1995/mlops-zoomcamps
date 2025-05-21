import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pickle
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
# print(sklearn.__version__)

def readDataframe(filename):
    df_trips = pd.read_parquet(filename)
    # print(df_trips.dtypes)
    df_trips['trip_duration'] = df_trips.lpep_dropoff_datetime - df_trips.lpep_pickup_datetime
    df_trips.trip_duration = df_trips.trip_duration.apply(lambda td: td.total_seconds() / 60)

    # print(len(df_trips))

    # print(df_trips.trip_duration.describe())
    
    df_trips= df_trips[(df_trips.trip_duration >= 1) & (df_trips.trip_duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']

    df_trips[categorical] = df_trips[categorical].astype(str)
    
    return df_trips

df_train = readDataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-02.parquet')
df_valid = readDataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-01.parquet')

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']
dict_train = (df_train[categorical + numerical].to_dict(orient='records'))
dict_valid = (df_valid[categorical + numerical].to_dict(orient='records'))

print(len(dict_valid))
dv = DictVectorizer()
x_train = dv.fit_transform(dict_train)
x_valid = dv.transform(dict_valid)

# print(dv.feature_names_)

target = 'trip_duration'
y_train = df_train[target].values
y_valid = df_valid[target].values

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_valid)

mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
print('Linear Regrssion rsme: ', rmse)

with open('./models/len_reg.pin','wb') as f_out:
    pickle.dump((dv, lr), f_out)
    
# try Lasso Regression Model 
ls = Lasso(alpha=0.0001)
ls.fit(x_train, y_train)
y_pred = ls.predict(x_valid)

mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
print('Lasso rsme: ', rmse)

# try Ridge Regression Model 
ridge = Ridge(alpha=0.0001)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_valid)

mse = mean_squared_error(y_valid, y_pred)
rmse = mse ** 0.5
print('Ridge rsme: ', rmse)

y_pred = pd.Series(y_pred)
y_train = pd.Series(y_train)

plt.hist(y_pred, alpha=0.5, label="Prediction")
plt.hist(y_train, alpha=0.5, label="Actual")
plt.legend()
plt.show()

