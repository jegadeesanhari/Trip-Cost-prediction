# Importing Packages
import pickle
import numpy as np 
import pandas as pd 
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Reading data from local directory
df_train =  pd.read_csv(
    filepath_or_buffer = 'data/yellow_tripdata.csv',
    nrows=10000, 
    parse_dates=["tpep_pickup_datetime","tpep_dropoff_datetime"]
)

def add_new_date_time_features(dataset):
    dataset['hour'] = dataset.tpep_pickup_datetime.dt.hour
    dataset['day'] = dataset.tpep_pickup_datetime.dt.day
    dataset['month'] = dataset.tpep_pickup_datetime.dt.month
    dataset['year'] = dataset.tpep_pickup_datetime.dt.year
    dataset['day_of_week'] = dataset.tpep_pickup_datetime.dt.dayofweek
    return dataset

dataset = add_new_date_time_features(df_train)

dataset.fare_amount[(dataset.fare_amount <= 0) | (dataset.fare_amount >= 350)].count()

print(f'Row count before elimination - {dataset.shape[0]}')
dataset = dataset[dataset.fare_amount.between(0, 350, inclusive='neither')]
print(f'Row count after elimination - {dataset.shape[0]}')

dataset.passenger_count[(dataset.passenger_count < 1) | (dataset.passenger_count > 8)].count()

print(f'Row count before elimination - {dataset.shape[0]}')
dataset = dataset[dataset.passenger_count.between(0, 8, inclusive='neither')]
print(f'Row count after elimination - {dataset.shape[0]}')

dataset["store_and_fwd_flag"] = dataset["store_and_fwd_flag"].astype('category')
dataset["store_and_fwd_flag"] = dataset["store_and_fwd_flag"].cat.codes

selected_predictors = [
    'VendorID', 
    'passenger_count', 
    'trip_distance', 
    'RatecodeID',
    'store_and_fwd_flag',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'hour',
    'month',
    'year',
    'day_of_week'
]

X = dataset.loc[:, selected_predictors].values
y = dataset['fare_amount'].values

# using the train test split function 
X_train, X_test, y_train, y_test = train_test_split(
                          X, y,  
                          random_state=104, 
                          train_size=0.8, 
                          shuffle=True
)

rand_forest_regressor = RandomForestRegressor()
rand_forest_regressor.fit(X_train, y_train)

y_rand_forest_predict = rand_forest_regressor.predict(X_test)

random_forest_mae = mean_absolute_error(y_test,y_rand_forest_predict)
random_forest_mse = mean_squared_error(y_test,y_rand_forest_predict)
random_forest_rmsel = np.log(np.sqrt(mean_squared_error(y_test,y_rand_forest_predict)))
random_forest_r2 = r2_score(y_test,y_rand_forest_predict)
print(f'=============')
print(f'Random Forest')
print(f'=============')
print(f'Random Forest Mean Squared Error - {random_forest_mse}')
print(f'Random Forest Mean Absolute Error - {random_forest_mae}')
print(f'Random Forest Root Mean Squared Log Error - {random_forest_rmsel}')
print(f'Random Forest R Squared Error - {random_forest_r2}')

RM_final_score = {
    "Metrics": ["RF_MSE", "RF_MAE", "RF_RMSE", "RF_R2"],
    "Score": [random_forest_mse, random_forest_mae, random_forest_rmsel, random_forest_r2]
}
df_RM_score = pd.DataFrame(RM_final_score)

XGB_model = XGBRegressor(learning_rate=0.3, max_depth=6, n_estimators=500)
XGB_model.fit(X_train, y_train)

y_XGB_predict = XGB_model.predict(X_test)

XGB_model_mae = mean_absolute_error(y_test,y_XGB_predict)
XGB_model_mse = mean_squared_error(y_test,y_XGB_predict)
XGB_model_rmsel = np.log(np.sqrt(mean_squared_error(y_test,y_XGB_predict)))
XGB_model_r2 = r2_score(y_test,y_XGB_predict)

print(f'=======')
print(f'XGBoost')
print(f'=======')
print(f'XGBoost Mean Squared Error - {XGB_model_mse}')
print(f'XGBoost Mean Absolute Error - {XGB_model_mae}')
print(f'XGBoost Root Mean Squared Log Error - {XGB_model_rmsel}')
print(f'XGBoost R Squared Error - {XGB_model_r2}')

XGB_final_score = {
    "Metrics": ["XGB_MSE", "XGB_MAE", "XGB_RMSE", "XGB_R2"],
    "Score": [XGB_model_mse, XGB_model_mae, XGB_model_rmsel, XGB_model_r2]
}
df_XGB_score = pd.DataFrame(XGB_final_score)

lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=35, n_estimators=300)
lgb_model.fit(X_train, y_train)

y_LGB_predict = lgb_model.predict(X_test)

LGB_model_mae = mean_absolute_error(y_test,y_LGB_predict)
LGB_model_mse = mean_squared_error(y_test,y_LGB_predict)
LGB_model_rmsel = np.log(np.sqrt(mean_squared_error(y_test,y_LGB_predict)))
LGB_model_r2 = r2_score(y_test,y_LGB_predict)

print(f'=============')
print(f'LGBMRegressor')
print(f'=============')
print(f'LGBMRegressor Mean Squared Error - {LGB_model_mse}')
print(f'LGBMRegressor Mean Absolute Error - {LGB_model_mae}')
print(f'LGBMRegressor Root Mean Squared Log Error - {LGB_model_rmsel}')
print(f'LGBMRegressor R Squared Error - {LGB_model_r2}')

LGB_final_score = {
    "Metrics": ["LGB_MSE", "LGB_MAE", "LGB_RMSE", "LGB_R2"],
    "Score": [LGB_model_mse, LGB_model_mae, LGB_model_rmsel, LGB_model_r2]
}
df_LGB_score = pd.DataFrame(LGB_final_score)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)
y_lr_predict = reg.predict(X_test)

LR_model_mae = mean_absolute_error(y_test,y_lr_predict)
LR_model_mse = mean_squared_error(y_test,y_lr_predict)
LR_model_rmsel = np.log(np.sqrt(mean_squared_error(y_test,y_lr_predict)))
LR_model_r2 = r2_score(y_test,y_lr_predict)

print(f'================')
print(f'LinearRegression')
print(f'================')
print(f'LinearRegression Mean Squared Error - {LR_model_mse}')
print(f'LinearRegression Mean Absolute Error - {LR_model_mae}')
print(f'LinearRegression Root Mean Squared Log Error - {LR_model_rmsel}')
print(f'LinearRegression R Squared Error - {LR_model_r2}')

LR_final_score = {
    "Metrics": ["LR_MSE", "LR_MAE", "LR_RMSE", "LR_R2"],
    "Score": [LR_model_mse, LR_model_mae, LR_model_rmsel, LR_model_r2]
}
df_LR_score = pd.DataFrame(LR_final_score)

df_score = pd.concat([df_RM_score, df_XGB_score, df_LGB_score, df_LR_score], axis=0, ignore_index=True)
print(df_score)

pickle_out = open("models/rand_forest_regressor.pkl", "wb") 
pickle.dump(rand_forest_regressor, pickle_out)

pickle_out = open("models/XGB_model.pkl", "wb") 
pickle.dump(XGB_model, pickle_out) 

pickle_out = open("models/lgb_model.pkl", "wb") 
pickle.dump(lgb_model, pickle_out) 

pickle_out = open("models/reg_model.pkl", "wb") 
pickle.dump(reg, pickle_out) 

pickle_out.close()