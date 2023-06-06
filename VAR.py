import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_inflow.csv')


dataset = df.iloc[:,0:5]

dataset.drop("usd_rate", axis =1, inplace = True)
dataset['TRAN_DATE']= pd.to_datetime(dataset['TRAN_DATE'])
dataset.index = dataset['TRAN_DATE']
del dataset['TRAN_DATE']

result = seasonal_decompose(dataset['Amount_in_USD'])

trend = result.trend
seasonal = result.seasonal
residuals = result.resid

dataset['Amount_in_USD'] = dataset['Amount_in_USD']-trend -seasonal
dataset.dropna(inplace = True)

dataset.iloc[round(dataset.shape[0]*0.8)]

train_timeseries = dataset.iloc[:round(dataset.shape[0]*0.8),:]
test_timeseries = dataset.iloc[round(dataset.shape[0]*0.8):,:]

aic = []
for i in [1,2,3,4,5,6,7,8,9,10]:
    model = VAR(train_timeseries)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    aic.append(results.aic)
    
result = model.fit(round(min(aic)))
print(result.summary() )

      
pred = result.forecast(result.endog, steps =len(test_timeseries))
# pred_train = result.forecast(result.endog, steps = len(train_timeseries))

pred_df = pd.DataFrame(pred ,index = test_timeseries.index, columns = test_timeseries.columns )
# pred_df_train = pd.DataFrame(pred_train, index = train_timeseries.index, columns  = train_timeseries.columns)

forecast = pred_df['Amount_in_USD']

test_values = test_timeseries['Amount_in_USD']+trend.iloc[-len(test_timeseries):]+seasonal.iloc[-len(test_timeseries):]
predicted_values = forecast+trend.iloc[-len(test_timeseries):]+seasonal.iloc[-len(test_timeseries):]
train_values = train_timeseries['Amount_in_USD'] +trend.iloc[:len(train_timeseries)]+seasonal.iloc[:len(train_timeseries)]

plt.figure(figsize=(12,6))
plt.plot(train_timeseries.index, train_values[3:], label = 'Train')
plt.plot(test_timeseries.index, test_values[:-3], label = 'Test')
plt.plot(pred_df.index, predicted_values[:-3], label = 'Predicted')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Amount in USD")
plt.title("Inflow Forecast using VAR model")
plt.show()

mse = mean_squared_error(test_values.dropna(), predicted_values.dropna())
print(f"mean squared error : {mse :.4f}")
print("RMSE :", np.sqrt(mse))
print("MPE :", mean_absolute_percentage_error(test_values.dropna(), predicted_values.dropna()))
