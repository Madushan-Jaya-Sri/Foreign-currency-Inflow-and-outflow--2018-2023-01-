import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error


time_series_inflow_dataset = pd.read_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_inflow.csv')

dataset = time_series_inflow_dataset.copy()
# Convert date to datetime format
dataset['TRAN_DATE'] = pd.to_datetime(dataset['TRAN_DATE'])

# Sort the dataset by date
dataset.sort_values(by=['TRAN_DATE'], inplace=True)

# Set the date as the index
dataset.set_index('TRAN_DATE', inplace=True)
#del dataset['usd_rate']

# Load the data
df = dataset

# Decompose the data into trend, seasonal, and residual components
decomposition = seasonal_decompose(df['Amount_in_USD'], model='additive', period=11)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Deseasonalize the data
deseasonalized = df['Amount_in_USD'] - seasonal

# Detrend the data
detrended = deseasonalized - trend
#detrended = deseasonalized.diff().dropna()

df['Amount_in_USD'] = detrended

df.dropna(inplace=True)
df.columns

timeseries_data = df
train_size=round(timeseries_data.shape[0]*0.8)

train_timeseries = timeseries_data.iloc[:train_size,:]
test_timeseries = timeseries_data.iloc[train_size:,:]

timeseries_data = train_timeseries.copy()

scaler = StandardScaler()
scaler = scaler.fit(timeseries_data)
timeseries_data_scaled = scaler.transform(timeseries_data)


trainX=[]
trainY=[]


n_future = 30
n_past = 11

for i in range(n_past, len(timeseries_data_scaled)-n_future+1):
    trainX.append(timeseries_data_scaled[i-n_past:i, 0:timeseries_data.shape[1]])
    trainY.append(timeseries_data_scaled[i+n_future - 1:i + n_future,0])
    
trainX,trainY = np.array(trainX),np.array(trainY)

  
trainX,trainY = np.array(trainX),np.array(trainY)

print('trainX shape == {}'.format(trainX.shape))
print('trainY shape == {}'.format(trainY.shape))


model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(GRU(units=50, return_sequences=True))
model.add(GRU(units=40, return_sequences=True))
model.add(GRU(units=50))
model.add(Dense(units=1))


model.compile(optimizer = 'adam', loss= 'mse')
model.summary()
es = EarlyStopping(monitor= 'val_loss', patience= 10 )

history = model.fit(trainX,trainY,epochs=100,batch_size = 200, validation_split=0.1, verbose = 1, callbacks = [es])
plt.title("Model Loss curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'],label = 'Validation loss')
plt.legend()
plt.show()

n_future = len(test_timeseries)

forecast = model.predict(trainX[-n_future:])

forecast_copies = np.repeat(forecast, timeseries_data.shape[1],axis=1)
y_pred = scaler.inverse_transform(forecast_copies)[:,0]

#forecast_period = pd.date_range('12-01-2022','02-28-2023').to_list()
forecast_period = pd.date_range(list(timeseries_data.index)[-1],periods = n_future, freq ='1d').tolist()
forecasted_data = pd.DataFrame(index = forecast_period)
forecasted_data.dtypes

forecasted_data['Predicted amount']= y_pred + trend[-n_future:]+seasonal[-n_future:]

plt.plot(timeseries_data['Amount_in_USD']+trend[-len(timeseries_data):]+
         seasonal[-len(timeseries_data):],label = 'Training series')
plt.plot(test_timeseries['Amount_in_USD']+trend[-len(test_timeseries):]+
         seasonal[-len(test_timeseries):],label = 'Testing series')
plt.plot(forecasted_data,label = 'Forecasted series')
plt.title('GRU - Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount (USD)')

plt.legend()
plt.show()

test_values = test_timeseries['Amount_in_USD']+trend[-len(test_timeseries):]+seasonal[-len(test_timeseries):]
print("RMSE : ",  np.sqrt(mean_squared_error(test_values[-len(trend[-len(test_timeseries):]):].fillna(0),
                                             forecasted_data[:len(test_values)].fillna(0)))) 




