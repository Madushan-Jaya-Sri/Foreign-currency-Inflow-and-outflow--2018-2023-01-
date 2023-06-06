import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

time_series_inflow_dataset = pd.read_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_inflow.csv')

# =============================================================================
# # # # # INFLOW
# =============================================================================
time_series_inflow_dataset.index = time_series_inflow_dataset['TRAN_DATE']
dataset = time_series_inflow_dataset.copy()

sns.heatmap(dataset.drop('usd_rate',axis =1).corr())



# read in your dataset as a pandas DataFrame
df = dataset.drop(["usd_rate"],axis =1)

# create a design matrix with the independent variables
X = df[df.columns]

# add a constant to the design matrix (required for statsmodels)
X = add_constant(X)

# calculate the VIF values for each variable
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# print the VIF values
print(vif)




#dataset['Amount_in_USD'] = np.log(dataset['Amount_in_USD'])
# =============================================================================
# checking stationarity
# =============================================================================


# Apply ADF test
result = adfuller(dataset['Amount_in_USD'])

# Create a table to display the results
adf_table = pd.DataFrame({'ADF Test Statistic': [result[0]],
                          'p-value': [result[1]],
                          '1% Critical Value': [result[4]['1%']],
                          '5% Critical Value': [result[4]['5%']],
                          '10% Critical Value': [result[4]['10%']]})

# Display the table
print(adf_table)

# Apply KPSS test
result_kpss = kpss(dataset['Amount_in_USD'])

# Create a table to display the results
kpss_table = pd.DataFrame({'KPSS Test Statistic': [result_kpss[0]],
                          'p-value': [result_kpss[1]],
                          '10% Critical Value': [result_kpss[3]['10%']],
                          '5% Critical Value': [result_kpss[3]['5%']],
                          '2.5% Critical Value': [result_kpss[3]['2.5%']],
                          '1% Critical Value': [result_kpss[3]['1%']]})

# Display the table
print(kpss_table)

# =============================================================================
# time series components
# =============================================================================
dataset.index = pd.to_datetime(dataset.index)

seasonal_decompose(dataset['Amount_in_USD']).plot()
plt.show()


# Perform seasonal decomposition
result = seasonal_decompose(dataset['Amount_in_USD'], model='additive', period=12)

# Plot the seasonal component
result.seasonal.plot()
plt.show()


seasonal_period = 1
# Perform seasonal decomposition
result = seasonal_decompose(dataset['Amount_in_USD'], model='additive', period=seasonal_period)

# Plot the trend component
result.trend.plot()
plt.show()

# Print the first few values of the trend series
print(result.trend.head())

# =============================================================================
# making series stationary
# =============================================================================

# Load your time series data
df = dataset

# Perform seasonal decomposition
result = seasonal_decompose(df['Amount_in_USD'], model='additive', period=11)

# Extract the trend, seasonal, and residual components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Remove the trend and seasonal pattern from the original data
detrended = df['Amount_in_USD'] - trend
deseasonalized = detrended - seasonal

# Plot the original data, trend, seasonal, and residual components
result.plot()
plt.show()

# Plot the detrended and deseasonalized data
deseasonalized.plot()
plt.title("Stationary time series of inflow amount in USD")
plt.xlabel("Date")
plt.ylabel("Amount in USD")
plt.show()

re = seasonal_decompose(deseasonalized.dropna(), model='additive', period=11)
re.plot()

diff_data = df['Amount_in_USD'].diff().dropna()
plt.plot(diff_data)

# =============================================================================
# rechecking the stationarity
# =============================================================================
result_kpss = kpss(deseasonalized.dropna())

# Create a table to display the results
kpss_table = pd.DataFrame({'KPSS Test Statistic': [result_kpss[0]],
                          'p-value': [result_kpss[1]],
                          '10% Critical Value': [result_kpss[3]['10%']],
                          '5% Critical Value': [result_kpss[3]['5%']],
                          '2.5% Critical Value': [result_kpss[3]['2.5%']],
                          '1% Critical Value': [result_kpss[3]['1%']]})

# Display the table
print(kpss_table)

# Apply ADF test
result_ADF = kpss(diff_data)

# Create a table to display the results
adf_table = pd.DataFrame({'ADF Test Statistic': [result_ADF[0]],
                          'p-value': [result_ADF[1]],
                          '1% Critical Value': [result_ADF[3]['1%']],
                          '5% Critical Value': [result_ADF[3]['5%']],
                          '10% Critical Value': [result_ADF[3]['10%']]})

# Display the table
print(adf_table)
 
# =============================================================================
# Scalling data
# =============================================================================

dataset = time_series_inflow_dataset.copy()
scle = StandardScaler()

dataset['usd_rate'] = scle.fit_transform(dataset['usd_rate'].values.reshape(-1,1))
dataset['M_inf_rate'] = scle.fit_transform(dataset['M_inf_rate'].values.reshape(-1,1))
dataset['AVG(T.FDINT)'] = scle.fit_transform(dataset['AVG(T.FDINT)'].values.reshape(-1,1))
#dataset['Amount_in_USD'] = scle.fit_transform(dataset['Amount_in_USD'].values.reshape(-1,1))

# =============================================================================
# ACF & PACF plots
# =============================================================================
fig , ax = plt.subplots(1,2, figsize = (20,5))

plot_acf(deseasonalized.dropna(), lags = 15,ax=ax[0])
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.show()
plot_pacf(deseasonalized.dropna(), lags = 15, method = "ols",ax=ax[1])
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.show()
'''
p = 1
d = 1
q = 1
'''
# =============================================================================
# ARIMA - forecasting
# =============================================================================


#df_usd = dataset['Amount_in_USD'].resample('M').sum()


#df_usd = np.log(dataset['Amount_in_USD'])
#df_usd = dataset['Amount_in_USD']
df_usd = deseasonalized.dropna()


train_size  = int(len(df_usd)*0.8)
train, test = df_usd [: train_size], df_usd [train_size:]



model = ARIMA(train , order =( 1,1,1))
model_fit =  model.fit()

predictions = model_fit.predict(start = len (train), end = len( train)+len(test)-1 , typ ='levels')
predictions_train  = model_fit.predict(start = 1, end = len( train)-1 , typ ='levels')


predictions_train.index= pd.to_datetime(predictions_train.index)


predictions.index= pd.to_datetime(predictions.index)
test.index= pd.to_datetime(test.index)
train.index = pd.to_datetime(train.index)


# =============================================================================
# predicting for 30 days ahead 
# =============================================================================
future_period = 90
forecast_df = pd.DataFrame()
future_pred = model_fit.predict(start = len (df_usd), end = len(df_usd) + future_period, typ ='levels')
future_values = future_pred.values+trend.iloc[-future_period-1:]+seasonal.iloc[-future_period-1:] 

predic_train = predictions_train.values+trend.iloc[:len(train)-1]+seasonal.iloc[:len(train)-1] 



# Create a new time index that spans the next 12 months from T
time_index = pd.date_range(start=df_usd.index[-1], periods=future_period+1, freq='d')

future_values.index = time_index

plt.title('ARIMA - Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')

train_values = train+trend.iloc[: train_size]+seasonal.iloc[: train_size]
test_values = test+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = predictions+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.plot(train_values, label = 'Trainned values')
plt.plot(test_values, label ='Test values')
plt.plot(predicted_values, label ='Predicted')
plt.legend()
plt.show()

plt.plot(train_values, label = 'Trainned values')
plt.plot(test_values, label ='Test values')
plt.plot(predic_train, label ='Predicted')
plt.legend()
plt.show()

plt.plot(future_values, label ='future values')
plt.legend()
plt.show()


MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))

print(model_fit.summary())

MSE = mean_squared_error(train_values.dropna()[:-1], predic_train.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(train_values.dropna()[:-1], predic_train.dropna()))

# =============================================================================
# model validataion and residual diagnosis
# =============================================================================
# Model Validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# define the number of folds for cross validation
n_splits = 5

# define the size of each fold as a percentage of the total dataset size
fold_size = int(len(df_usd) / n_splits)

# create the time series split object
tscv = TimeSeriesSplit(n_splits=n_splits)

# initialize an empty list to store the RMSE values for each fold
rmse_values = []

# initialize an empty dataframe to store the results
results_df = pd.DataFrame(columns=["Fold", "RMSE"])

# loop over the folds in the time series split object
for i, (train_index, test_index) in enumerate(tscv.split(df_usd)):
    
    # split the data into train and test sets for this fold
    train_data = df_usd[train_index]
    test_data = df_usd[test_index]
    
    # train the model on the training set for this fold
    model = ARIMA(train_data, order=(1,1,1))
    model_fit = model.fit()
    
    # make predictions on the test set for this fold
    predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')
    
    # calculate the RMSE for this fold and append it to the list of RMSE values
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    rmse_values.append(rmse)
    
    # add the results for this fold to the dataframe
    results_df = results_df.append({"Fold": i+1, "RMSE": rmse}, ignore_index=True)

# print the results dataframe
print(results_df)

# print the mean and standard deviation of the RMSE values across all folds
print("Mean RMSE:", np.mean(rmse_values))
print("Standard deviation of RMSE:", np.std(rmse_values))



# =============================================================================
# Residual Diagnosis
# =============================================================================
residuals = model_fit.resid
plt.plot(residuals)
plt.title('Residual Plot')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

acf = plot_acf(residuals, lags=20)
pacf = plot_pacf(residuals, lags=20)

MAPE = mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna())
RMSE = np.sqrt(mean_squared_error(test_values.dropna(), predicted_values.dropna()))
print("MAPE:", MAPE)
print("RMSE:", RMSE)

residuals = model_fit.resid
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
from statsmodels.graphics.gofplots import qqplot

qqplot(residuals, line='s')
plt.title('Normal Probability Plot')
plt.show()


from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
if dw < 2:
    print("Positive autocorrelation in residuals")
elif dw > 2:
    print("Negative autocorrelation in residuals")
else:
    print("No autocorrelation in residuals")



# =============================================================================
# Auto - ARIMA
# =============================================================================

scle = StandardScaler()

dataset['usd_rate'] = scle.fit_transform(dataset['usd_rate'].values.reshape(-1,1))
dataset['M_inf_rate'] = scle.fit_transform(dataset['M_inf_rate'].values.reshape(-1,1))
dataset['AVG(T.FDINT)'] = scle.fit_transform(dataset['AVG(T.FDINT)'].values.reshape(-1,1))
#dataset['Amount_in_USD'] = scle.fit_transform(dataset['Amount_in_USD'].values.reshape(-1,1))

dataset['Amount_in_USD']=deseasonalized
dataset.dropna(inplace = True)
dataset.drop("usd_rate",axis =1, inplace =True)
dataset.iloc[round(dataset.shape[0]*0.8)]

train_timeseries = dataset.iloc[:round(dataset.shape[0]*0.8),:]
test_timeseries = dataset.iloc[round(dataset.shape[0]*0.8):,:]

from pmdarima.arima import auto_arima

arima_model = auto_arima(train_timeseries['Amount_in_USD'],seasonal_period= 11,
                         exogenous = train_timeseries.iloc[:,1:]
                         ,trace = True)
arima_model.order

forecast = arima_model.predict(n_periods = len(test_timeseries),exogenous = test_timeseries.iloc[:,1:])

plt.plot(test_timeseries['Amount_in_USD'],label = 'Testing series')
plt.plot(forecast,label = 'Forecasted series')
plt.plot(train_timeseries['Amount_in_USD'],label = 'Training series')


test_values = test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.plot(train_timeseries['Amount_in_USD']+trend.iloc[: train_size]+seasonal.iloc[: train_size], label = 'Trainned values')
plt.plot(test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:], label ='Test values')
plt.plot(forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:] , label ='Predicted')

plt.legend()

plt.title('Auto - ARIMA - Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')
plt.show()

print(arima_model.summary())
#%matplotlib qt
MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))



# =============================================================================
# ARIMAX - forecasting
# =============================================================================

dataset = time_series_inflow_dataset.copy()


dataset['Amount_in_USD']=deseasonalized
dataset.dropna(inplace = True)
dataset.drop("usd_rate", axis = 1, inplace =True)
dataset.iloc[round(dataset.shape[0]*0.8)]

train_timeseries = dataset.iloc[:round(dataset.shape[0]*0.8),:]
test_timeseries = dataset.iloc[round(dataset.shape[0]*0.8):,:]


sarimax_model = SARIMAX(train_timeseries['Amount_in_USD'], seasonal_order=(1,1,1,11), trend='c',
                        order=(1, 1, 1), exog=train_timeseries.iloc[:, 1:])

res = sarimax_model.fit(disp=True)

res.summary()

forecast = res.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])

forecasted = res.predict(start = test_timeseries.index[0],
                         end = test_timeseries.index[-1],
                         exog = test_timeseries.iloc[:,1:])

train_values = train_timeseries['Amount_in_USD']+trend.iloc[: train_size]+seasonal.iloc[: train_size]
test_values = test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.title('ARIMAX - Forecasting')

plt.plot(train_values, label = 'Trainned values')
plt.plot(test_values, label ='Test values')
plt.plot(predicted_values, label ='Predicted')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')
plt.legend()
plt.show()


MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))

print(res.summary())

# =============================================================================
# 
# 
# # =============================================================================
# # Grid search
# # =============================================================================
# import itertools
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# 
# # Define the p, d, and q parameters to take any value between 0 and 2
# p = d = q = range(0, 3)
# 
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
# 
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 11) for x in list(itertools.product(p, d, q))]
# 
# # Create a list to store the results
# results = []
# 
# # Loop through all the parameter combinations
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             # Build SARIMAX model for the current combination of parameters
#             model = SARIMAX(train_timeseries['Amount_in_USD'], order=param, seasonal_order=param_seasonal, exog=train_timeseries.iloc[:, 1:])
#             # Fit the model
#             model_fit = model.fit(disp=False)
#             # Get the predicted values
#             predicted = model_fit.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])
#             # Calculate RMSE and add to the results list
#             rmse = sqrt(mean_squared_error(test_timeseries['Amount_in_USD'], predicted))
#             results.append((param, param_seasonal, rmse))
#         except:
#             continue
# 
# # Find the parameters with the lowest RMSE
# best_params = min(results, key=lambda x:x[2])
# print('Best Parameters:', best_params)
# 
# # Build the SARIMAX model with the best parameters
# model = SARIMAX(train_timeseries['Amount_in_USD'], order=best_params[0], seasonal_order=best_params[1], exog=train_timeseries.iloc[:, 1:])
# # Fit the model
# model_fit = model.fit(disp=False)
# 
# # Get the predicted values
# predicted = model_fit.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])
# 
# # Calculate RMSE
# rmse = sqrt(mean_squared_error(test_timeseries['Amount_in_USD'], predicted))
# print('RMSE:', rmse)
# 
# # Plot the predicted values against the actual values
# plt.plot(test_timeseries['Amount_in_USD'], label='Actual')
# plt.plot(predicted, label='Predicted')
# plt.legend()
# plt.show()
# 
# 
# 
# =============================================================================


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # OUTFLOW
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

time_series_outflow_dataset = pd.read_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_outflow.csv')



dataset = time_series_outflow_dataset.copy()

# =============================================================================
# checking stationarity
# =============================================================================


# Apply ADF test
result = adfuller(dataset['Amount_in_USD'])

# Create a table to display the results
adf_table = pd.DataFrame({'ADF Test Statistic': [result[0]],
                          'p-value': [result[1]],
                          '1% Critical Value': [result[4]['1%']],
                          '5% Critical Value': [result[4]['5%']],
                          '10% Critical Value': [result[4]['10%']]})

# Display the table
print(adf_table)

     
     
'''
series is stationary according to the adf test
'''     



# Apply KPSS test
result_kpss = kpss(dataset['Amount_in_USD'])

# Create a table to display the results
kpss_table = pd.DataFrame({'KPSS Test Statistic': [result_kpss[0]],
                          'p-value': [result_kpss[1]],
                          '10% Critical Value': [result_kpss[3]['10%']],
                          '5% Critical Value': [result_kpss[3]['5%']],
                          '2.5% Critical Value': [result_kpss[3]['2.5%']],
                          '1% Critical Value': [result_kpss[3]['1%']]})

# Display the table
print(kpss_table)

     


# =============================================================================
# time series components
# =============================================================================
dataset.index = pd.to_datetime(dataset.index)

seasonal_decompose(dataset['Amount_in_USD']).plot()
plt.show()


# Perform seasonal decomposition
result = seasonal_decompose(dataset['Amount_in_USD'], model='additive', period=12)

# Plot the seasonal component
result.seasonal.plot()
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

seasonal_period = 1
# Perform seasonal decomposition
result = seasonal_decompose(dataset['Amount_in_USD'], model='additive', period=seasonal_period)

# Plot the trend component
result.trend.plot()
plt.show()

# Print the first few values of the trend series
print(result.trend.head())

# =============================================================================
# making series stationary
# =============================================================================

# Load your time series data
df = dataset

# Perform seasonal decomposition
result = seasonal_decompose(df['Amount_in_USD'], model='additive', period=11)

# Extract the trend, seasonal, and residual components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Remove the trend and seasonal pattern from the original data
detrended = df['Amount_in_USD'] - trend
deseasonalized = detrended - seasonal

# Plot the original data, trend, seasonal, and residual components
result.plot()
plt.show()

# Plot the detrended and deseasonalized data
deseasonalized.plot()
plt.title("Stationary time series of inflow amount in USD")
plt.xlabel("Date")
plt.ylabel("Amount in USD")
plt.show()

re = seasonal_decompose(deseasonalized.dropna(), model='additive', period=11)
re.plot()

diff_data = df['Amount_in_USD'].diff().dropna()
plt.plot(diff_data)

# =============================================================================
# rechecking the stationarity
# =============================================================================
result_kpss = kpss(deseasonalized.dropna())

# Create a table to display the results
kpss_table = pd.DataFrame({'KPSS Test Statistic': [result_kpss[0]],
                          'p-value': [result_kpss[1]],
                          '10% Critical Value': [result_kpss[3]['10%']],
                          '5% Critical Value': [result_kpss[3]['5%']],
                          '2.5% Critical Value': [result_kpss[3]['2.5%']],
                          '1% Critical Value': [result_kpss[3]['1%']]})

# Display the table
print(kpss_table)

# Apply ADF test
result_ADF = kpss(diff_data)

# Create a table to display the results
adf_table = pd.DataFrame({'ADF Test Statistic': [result_ADF[0]],
                          'p-value': [result_ADF[1]],
                          '1% Critical Value': [result_ADF[3]['1%']],
                          '5% Critical Value': [result_ADF[3]['5%']],
                          '10% Critical Value': [result_ADF[3]['10%']]})

# Display the table
print(adf_table)
 


# =============================================================================
# Scalling data
# =============================================================================

dataset = time_series_outflow_dataset.copy()

scle = StandardScaler()

dataset['usd_rate'] = scle.fit_transform(dataset['usd_rate'].values.reshape(-1,1))
dataset['M_inf_rate'] = scle.fit_transform(dataset['M_inf_rate'].values.reshape(-1,1))
dataset['AVG(T.FDINT)'] = scle.fit_transform(dataset['AVG(T.FDINT)'].values.reshape(-1,1))
#dataset['Amount_in_USD'] = scle.fit_transform(dataset['Amount_in_USD'].values.reshape(-1,1))





# =============================================================================
# ACF & PACF plots
# =============================================================================
fig , ax = plt.subplots(1,2, figsize = (20,5))

plot_acf(deseasonalized.dropna(), lags = 15,ax=ax[0])
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plot_pacf(deseasonalized.dropna(), lags = 15, method = "ols",ax=ax[1])
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.show()

'''
p = 1
d = 1
q = 1

'''
# =============================================================================
# ARIMA - forecasting
# =============================================================================


from statsmodels.tsa.arima.model import ARIMA

#df_usd = dataset['Amount_in_USD'].resample('M').sum()


#df_usd = np.log(dataset['Amount_in_USD'])
#df_usd = dataset['Amount_in_USD']
df_usd = deseasonalized.dropna()


train_size  = int(len(df_usd)*0.8)
train, test = df_usd [: train_size], df_usd [train_size:]



model = ARIMA(train , order =( 1,1,1))
model_fit =  model.fit()

predictions = model_fit.predict(start = len (train), end = len( train)+len(test)-1 , typ ='levels')


predictions.index= pd.to_datetime(predictions.index)
test.index= pd.to_datetime(test.index)
train.index = pd.to_datetime(train.index)


# =============================================================================
# predicting for 30 days ahead 
# =============================================================================
future_period = 90
forecast_df = pd.DataFrame()
future_pred = model_fit.predict(start = len (df_usd), end = len(df_usd) + future_period, typ ='levels')
future_values = future_pred.values+trend.iloc[-future_period-1:]+seasonal.iloc[-future_period-1:] 

# Create a new time index that spans the next 12 months from T
time_index = pd.date_range(start=df_usd.index[-1], periods=future_period+1, freq='d')

future_values.index = time_index

plt.title('ARIMA - Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')

train_values = train+trend.iloc[: train_size]+seasonal.iloc[: train_size]
test_values = test+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = predictions+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.plot(train_values, label = 'Trainned values')
plt.plot(test_values, label ='Test values')
plt.plot(predicted_values, label ='Predicted')
plt.legend()
plt.show()


plt.plot(future_values, label ='future values')
plt.legend()
plt.show()


MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))

print(model_fit.summary())



# =============================================================================
# Auto - ARIMA
# =============================================================================


dataset = time_series_outflow_dataset.copy()


scle = StandardScaler()

dataset['usd_rate'] = scle.fit_transform(dataset['usd_rate'].values.reshape(-1,1))
dataset['M_inf_rate'] = scle.fit_transform(dataset['M_inf_rate'].values.reshape(-1,1))
dataset['AVG(T.FDINT)'] = scle.fit_transform(dataset['AVG(T.FDINT)'].values.reshape(-1,1))
#dataset['Amount_in_USD'] = scle.fit_transform(dataset['Amount_in_USD'].values.reshape(-1,1))

dataset['Amount_in_USD']=deseasonalized
dataset.dropna(inplace = True)

dataset.iloc[round(dataset.shape[0]*0.8)]

train_timeseries = dataset.iloc[:round(dataset.shape[0]*0.8),:]
test_timeseries = dataset.iloc[round(dataset.shape[0]*0.8):,:]

from pmdarima.arima import auto_arima

arima_model = auto_arima(train_timeseries['Amount_in_USD'],seasonal_period= 11,
                         exogenous = train_timeseries.iloc[:,1:]
                         ,trace = True)
arima_model.order


#forecast = arima_model.predict(start = test_timeseries.index[0],end = test_timeseries.index[-1],exog = test_timeseries[['M_inf_rate','AVG(T.FDINT)']])
forecast = arima_model.predict(n_periods = len(test_timeseries),exogenous = test_timeseries.iloc[:,1:])

plt.plot(test_timeseries['Amount_in_USD'],label = 'Testing series')
plt.plot(forecast,label = 'Forecasted series')
plt.plot(train_timeseries['Amount_in_USD'],label = 'Training series')


test_values = test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.plot(train_timeseries['Amount_in_USD']+trend.iloc[: train_size]+seasonal.iloc[: train_size], label = 'Trainned values')
plt.plot(test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:], label ='Test values')
plt.plot(forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:] , label ='Predicted')

plt.legend()

plt.title('Auto - ARIMA - Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')
plt.show()

print(arima_model.summary())
#%matplotlib qt
MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))



# =============================================================================
# ARIMAX - forecasting
# =============================================================================

dataset = time_series_outflow_dataset.copy()

# =============================================================================
# scle = StandardScaler()
# 
# dataset['usd_rate'] = scle.fit_transform(dataset['usd_rate'].values.reshape(-1,1))
# dataset['M_inf_rate'] = scle.fit_transform(dataset['M_inf_rate'].values.reshape(-1,1))
# dataset['AVG(T.FDINT)'] = scle.fit_transform(dataset['AVG(T.FDINT)'].values.reshape(-1,1))
# #dataset['Amount_in_USD'] = scle.fit_transform(dataset['Amount_in_USD'].values.reshape(-1,1))
# 
# 
# =============================================================================

dataset['Amount_in_USD']=deseasonalized
dataset.dropna(inplace = True)

dataset.iloc[round(dataset.shape[0]*0.8)]

train_timeseries = dataset.iloc[:round(dataset.shape[0]*0.8),:]
test_timeseries = dataset.iloc[round(dataset.shape[0]*0.8):,:]


sarimax_model = SARIMAX(train_timeseries['Amount_in_USD'], seasonal_order=(1,1,1,11), trend='c',
                        order=(1, 1, 1), exog=train_timeseries.iloc[:, 1:])

res = sarimax_model.fit(disp=True)

res.summary()

forecast = res.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])



forecasted = res.predict(start = test_timeseries.index[0],
                         end = test_timeseries.index[-1],
                         exog = test_timeseries.iloc[:,1:])

train_values = train_timeseries['Amount_in_USD']+trend.iloc[: train_size]+seasonal.iloc[: train_size]
test_values = test_timeseries['Amount_in_USD']+trend.iloc[train_size:]+seasonal.iloc[train_size:]
predicted_values = forecast+trend.iloc[train_size:]+seasonal.iloc[train_size:]

plt.title('ARIMAX - Forecasting')

plt.plot(train_values, label = 'Trainned values')
plt.plot(test_values, label ='Test values')
plt.plot(predicted_values, label ='Predicted')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')
plt.legend()
plt.show()


MSE = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print("RMSE :",np.sqrt(MSE))
print(mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))

print(res.summary())


# =============================================================================
# 
# # =============================================================================
# # Grid search
# # =============================================================================
# import itertools
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# 
# # Define the p, d, and q parameters to take any value between 0 and 2
# p = d = q = range(0, 3)
# 
# # Generate all different combinations of p, q and q triplets
# pdq = list(itertools.product(p, d, q))
# 
# # Generate all different combinations of seasonal p, q and q triplets
# seasonal_pdq = [(x[0], x[1], x[2], 11) for x in list(itertools.product(p, d, q))]
# 
# # Create a list to store the results
# results = []
# 
# # Loop through all the parameter combinations
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             # Build SARIMAX model for the current combination of parameters
#             model = SARIMAX(train_timeseries['Amount_in_USD'], order=param, seasonal_order=param_seasonal, exog=train_timeseries.iloc[:, 1:])
#             # Fit the model
#             model_fit = model.fit(disp=False)
#             # Get the predicted values
#             predicted = model_fit.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])
#             # Calculate RMSE and add to the results list
#             rmse = sqrt(mean_squared_error(test_timeseries['Amount_in_USD'], predicted))
#             results.append((param, param_seasonal, rmse))
#         except:
#             continue
# 
# # Find the parameters with the lowest RMSE
# best_params = min(results, key=lambda x:x[2])
# print('Best Parameters:', best_params)
# 
# # Build the SARIMAX model with the best parameters
# model = SARIMAX(train_timeseries['Amount_in_USD'], order=best_params[0], seasonal_order=best_params[1], exog=train_timeseries.iloc[:, 1:])
# # Fit the model
# model_fit = model.fit(disp=False)
# 
# # Get the predicted values
# predicted = model_fit.forecast(steps=len(test_timeseries), exog=test_timeseries.iloc[:, 1:])
# 
# # Calculate RMSE
# rmse = sqrt(mean_squared_error(test_timeseries['Amount_in_USD'], predicted))
# print('RMSE:', rmse)
# 
# # Plot the predicted values against the actual values
# plt.plot(test_timeseries['Amount_in_USD'], label='Actual')
# plt.plot(predicted, label='Predicted')
# plt.legend()
# plt.show()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================

