import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# # inflow dataset and inflow time series dataset
# =============================================================================
# =============================================================================

dataset = pd.read_csv("D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FC_IN.csv")

dataset.info()

dataset['TRAN_DATE'] =pd.to_datetime(dataset['TRAN_DATE'])
dataset['CLS_DATE'] = pd.to_datetime(dataset['CLS_DATE'])
dataset['DUE_DATE'] = pd.to_datetime(dataset['DUE_DATE']) 


dataset['TRAN_DATE'].min()
dataset['TRAN_DATE'].max()

######### removing unnecessary columns ################
dataset.dropna(how = 'all',axis =1, inplace = True)


null = dataset.isna().sum()
print('Null records: ',null)
duplicates = dataset.duplicated().sum()
print('Duplicate records: ',duplicates)
dataset.drop_duplicates(inplace= True)

date_var = []

for i in dataset.columns:        
        for k in range(0,len(dataset[i])):
            if ('/' in str(dataset[i][int(k)])) & ('/' in str(dataset[i][int(k+1)])):
                    date_var.append(i)
                    k=k+1
                    break
            else:
                break
            
print(date_var)

dataset['T_year'] = dataset['TRAN_DATE'].dt.strftime('%Y')
dataset['T_Month'] = dataset['TRAN_DATE'].dt.strftime('%m')
dataset['T_Day'] = dataset['TRAN_DATE'].dt.strftime('%d')



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

dataset = dataset[dataset['TRAN_DATE']>='2018-01-01']

# =============================================================================
# merging with exchange rate - dataset
# =============================================================================

dataset['CRRYEARMON'] = dataset['T_year'] + dataset['T_Month']

data2 = pd.read_csv('D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FCY_CURR_CONV.csv')

null = data2.isna().sum()
print('Null records: ',null)
duplicates = data2.duplicated().sum()
print('Duplicate records: ',duplicates)
data2.drop_duplicates(inplace= True)

data2['CRRYEARMON'] = data2['CRRYEARMON'].astype(str)

dataset.rename(columns={'TRAN_CRNCY_CODE':'CURRDESC'},inplace =True)

dataset_new = pd.merge(dataset, data2, how = "left", on = ['CURRDESC','CRRYEARMON'])   
dataset_new.dtypes

dataset_new['REV_RATE'].isna().sum()
dataset_new[dataset_new['REV_RATE'].isna()]

# =============================================================================
# converting all currencies to usd 
# =============================================================================


USD_RATE_DATA= data2[data2['CURRDESC']=='USD'][['REV_RATE','CRRYEARMON']]
USD_RATE_DATA['usd_rate']= USD_RATE_DATA['REV_RATE']
del USD_RATE_DATA['REV_RATE']
USD_RATE_DATA['CRRYEARMON'] = USD_RATE_DATA['CRRYEARMON'].astype(str)


dataset_new_with_USD = pd.merge(dataset_new, USD_RATE_DATA, how = "left", on = ['CRRYEARMON']) 

dataset_new_with_USD['Tran_in_LKR'] = dataset_new_with_USD['REV_RATE']*dataset_new_with_USD['TRAN_AMT']

dataset_new_with_USD['Amount_in_USD'] = dataset_new_with_USD['Tran_in_LKR']/dataset_new_with_USD['usd_rate']

dataset_new_with_USD.isna().sum()


# =============================================================================
# merging with interest rate
# =============================================================================

Int_rate = pd.read_csv('D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FC FD Int Rate.csv')

Int_rate.duplicated().sum()
Int_rate.drop_duplicates(inplace= True)

Int_rate['CRRYEARMON']=Int_rate['FDMONYEAR']
del Int_rate['FDMONYEAR']

Int_rate['CRRYEARMON'] = Int_rate['CRRYEARMON'].astype(str)


dataset_new_with_USD_and_intr = pd.merge(dataset_new_with_USD, Int_rate, how = "left", on = ['CRRYEARMON']) 

dataset_new_with_USD_and_intr.isna().sum()

# =============================================================================
# merging with inflation rate
# =============================================================================

inf_data = pd.read_excel('D:\\Projects\\14197- Madushan\\CAM_PRO\\Complete_Pro\\Inflation_rate.xlsx')


inf_data['CRRYEARMON']=inf_data['INF_Y_M']
del inf_data['INF_Y_M']

inf_data['CRRYEARMON'] = inf_data['CRRYEARMON'].astype(str)

dataset_new_with_USD_and_intr_infl = pd.merge(dataset_new_with_USD_and_intr, 
                                              inf_data, how = "left", on = ['CRRYEARMON']) 
dataset_new_with_USD_and_intr_infl.dtypes

dataset_new_with_USD_and_intr_infl.isna().sum()

# =============================================================================
# final full dataset
# =============================================================================

full_inflow_dataset = dataset_new_with_USD_and_intr_infl

full_inflow_dataset['TRAN_DATE']= pd.to_datetime(full_inflow_dataset['TRAN_DATE'])
DF_grouped =full_inflow_dataset.groupby(by= 'TRAN_DATE').agg(
    {'Amount_in_USD':'sum',
     'usd_rate':'mean',
     'M_inf_rate':'mean',
     'AVG(T.FDINT)':'mean',
     'BASLE_CODE_DESC':(lambda x: x.mode())}).reset_index()
# =============================================================================
# multiple mode values handling
# =============================================================================
for i, val in enumerate(DF_grouped['BASLE_CODE_DESC']):
    if isinstance(val, np.ndarray):
        if len(val) > 0:
            DF_grouped.loc[i, 'BASLE_CODE_DESC'] = val[0]
        else:
            DF_grouped.loc[i, 'BASLE_CODE_DESC'] = 'INDIVIDUALS'
  

# =============================================================================
# creating dates manually
# =============================================================================


dates= pd.date_range('01-01-2018','12-31-2022').to_list()
#dates= pd.date_range('01-01-2010','12-31-2022').to_list()
dates = pd.DataFrame(index=dates)
dates.reset_index(inplace=True)
dates['TRAN_DATE']=dates['index']
del dates['index']
time_series_inflow_dataset = pd.merge(dates,DF_grouped, how = 'left',on = 'TRAN_DATE')

time_series_inflow_dataset.isna().sum()

# =============================================================================
# filling null records
# =============================================================================

time_series_inflow_dataset.isna().sum()
time_series_inflow_dataset.fillna(method='ffill',inplace=True)
time_series_inflow_dataset.fillna(method='bfill',inplace=True)

time_series_inflow_dataset.dtypes

time_series_inflow_dataset.index = time_series_inflow_dataset['TRAN_DATE']
del time_series_inflow_dataset['TRAN_DATE']

# =============================================================================
# creating dummies for categorical variables
# =============================================================================

df_dummies = pd.get_dummies(time_series_inflow_dataset.select_dtypes('object'))

time_series_inflow_dataset  = pd.concat([time_series_inflow_dataset,df_dummies], axis = 1)

del time_series_inflow_dataset['BASLE_CODE_DESC']
time_series_inflow_dataset.to_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_inflow.csv')
full_inflow_dataset.to_csv('final_inflow_dataset.csv')




# =============================================================================
# =============================================================================
# # outflow dataset and outflow time series dataset
# =============================================================================
# =============================================================================

dataset = pd.read_csv("D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FC_OUT.csv")

dataset['TRAN_DATE'] =pd.to_datetime(dataset['TRAN_DATE'])
dataset['CLS_DATE'] = pd.to_datetime(dataset['CLS_DATE'])
dataset['DUE_DATE'] = pd.to_datetime(dataset['DUE_DATE']) 

dataset['TRAN_DATE'].min()
dataset['TRAN_DATE'].max()

######### removing unnecessary columns ################
dataset.dropna(how = 'all',axis =1, inplace = True)


null = dataset.isna().sum()
print('Null records: ',null)
duplicates = dataset.duplicated().sum()
print('Duplicate records: ',duplicates)
dataset.drop_duplicates(inplace= True)

date_var = []

for i in dataset.columns:        
        for k in range(0,len(dataset[i])):
            if ('/' in str(dataset[i][int(k)])) & ('/' in str(dataset[i][int(k+1)])):
                    date_var.append(i)
                    k=k+1
                    break
            else:
                break
            
print(date_var)

dataset['T_year'] = dataset['TRAN_DATE'].dt.strftime('%Y')
dataset['T_Month'] = dataset['TRAN_DATE'].dt.strftime('%m')
dataset['T_Day'] = dataset['TRAN_DATE'].dt.strftime('%d')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

dataset = dataset[dataset['TRAN_DATE']>='2018-01-01']


# =============================================================================
# merging with exchange rate - dataset
# =============================================================================

dataset['CRRYEARMON'] = dataset['T_year'] + dataset['T_Month']

data2 = pd.read_csv('D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FCY_CURR_CONV.csv')

null = data2.isna().sum()
print('Null records: ',null)
duplicates = data2.duplicated().sum()
print('Duplicate records: ',duplicates)
data2.drop_duplicates(inplace= True)

data2['CRRYEARMON'] = data2['CRRYEARMON'].astype(str)

dataset.rename(columns={'TRAN_CRNCY_CODE':'CURRDESC'},inplace =True)

dataset_new = pd.merge(dataset, data2, how = "left", on = ['CURRDESC','CRRYEARMON'])   
dataset_new.dtypes

dataset_new['REV_RATE'].isna().sum()
dataset_new[dataset_new['REV_RATE'].isna()]

# =============================================================================
# converting all currencies to usd 
# =============================================================================


USD_RATE_DATA= data2[data2['CURRDESC']=='USD'][['REV_RATE','CRRYEARMON']]
USD_RATE_DATA['usd_rate']= USD_RATE_DATA['REV_RATE']
del USD_RATE_DATA['REV_RATE']
USD_RATE_DATA['CRRYEARMON'] = USD_RATE_DATA['CRRYEARMON'].astype(str)


dataset_new_with_USD = pd.merge(dataset_new, USD_RATE_DATA, how = "left", on = ['CRRYEARMON']) 

dataset_new_with_USD['Tran_in_LKR'] = dataset_new_with_USD['REV_RATE']*dataset_new_with_USD['TRAN_AMT']

dataset_new_with_USD['Amount_in_USD'] = dataset_new_with_USD['Tran_in_LKR']/dataset_new_with_USD['usd_rate']

dataset_new_with_USD.isna().sum()


# =============================================================================
# merging with interest rate
# =============================================================================

Int_rate = pd.read_csv('D:\\Projects\\14197- Madushan\\CAM_PRO\\FC IN OUT\\FC FD Int Rate.csv')

Int_rate.duplicated().sum()
Int_rate.drop_duplicates(inplace= True)

Int_rate['CRRYEARMON']=Int_rate['FDMONYEAR']
del Int_rate['FDMONYEAR']

Int_rate['CRRYEARMON'] = Int_rate['CRRYEARMON'].astype(str)


dataset_new_with_USD_and_intr = pd.merge(dataset_new_with_USD, Int_rate, how = "left", on = ['CRRYEARMON']) 

dataset_new_with_USD_and_intr.isna().sum()

# =============================================================================
# merging with inflation rate
# =============================================================================

inf_data = pd.read_excel('D:\\Projects\\14197- Madushan\\CAM_PRO\\Complete_Pro\\Inflation_rate.xlsx')


inf_data['CRRYEARMON']=inf_data['INF_Y_M']
del inf_data['INF_Y_M']

inf_data['CRRYEARMON'] = inf_data['CRRYEARMON'].astype(str)

dataset_new_with_USD_and_intr_infl = pd.merge(dataset_new_with_USD_and_intr, inf_data, how = "left", on = ['CRRYEARMON']) 
dataset_new_with_USD_and_intr_infl.dtypes

dataset_new_with_USD_and_intr_infl.isna().sum()

# =============================================================================
# final full dataset
# =============================================================================

full_outflow_dataset = dataset_new_with_USD_and_intr_infl



full_outflow_dataset['TRAN_DATE']= pd.to_datetime(full_outflow_dataset['TRAN_DATE'])
DF_grouped =full_outflow_dataset.groupby(by= 'TRAN_DATE').agg(
    {'Amount_in_USD':'sum',
     'usd_rate':'mean',
     'M_inf_rate':'mean',
     'AVG(T.FDINT)':'mean',
     'BASLE_CODE_DESC':(lambda x: x.mode())}).reset_index()
# =============================================================================
# multiple mode values handling
# =============================================================================
for i, val in enumerate(DF_grouped['BASLE_CODE_DESC']):
    if isinstance(val, np.ndarray):
        if len(val) > 0:
            DF_grouped.loc[i, 'BASLE_CODE_DESC'] = val[0]
        else:
            DF_grouped.loc[i, 'BASLE_CODE_DESC'] = 'INDIVIDUALS'
  

# =============================================================================
# creating dates manually
# =============================================================================


dates= pd.date_range('01-01-2018','12-31-2022').to_list()
#dates= pd.date_range('01-01-2010','12-31-2022').to_list()
dates = pd.DataFrame(index=dates)
dates.reset_index(inplace=True)
dates['TRAN_DATE']=dates['index']
del dates['index']
time_series_outflow_dataset = pd.merge(dates,DF_grouped, how = 'left',on = 'TRAN_DATE')

time_series_outflow_dataset.isna().sum()

# =============================================================================
# filling null records
# =============================================================================

time_series_outflow_dataset.isna().sum()
time_series_outflow_dataset.fillna(method='ffill',inplace=True)
time_series_outflow_dataset.fillna(method='bfill',inplace=True)

time_series_outflow_dataset.dtypes

time_series_outflow_dataset.index = time_series_outflow_dataset['TRAN_DATE']
del time_series_outflow_dataset['TRAN_DATE']

# =============================================================================
# creating dummies for categorical variables
# =============================================================================

df_dummies = pd.get_dummies(time_series_outflow_dataset.select_dtypes('object'))

time_series_outflow_dataset  = pd.concat([time_series_outflow_dataset,df_dummies], axis = 1)

del time_series_outflow_dataset['BASLE_CODE_DESC']
time_series_outflow_dataset.to_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_outflow.csv')
full_outflow_dataset.to_csv('final_outflow_dataset.csv')




# =============================================================================
# checking the difference of inflow and outflow
# =============================================================================


plt.plot(time_series_outflow_dataset['Amount_in_USD']-time_series_inflow_dataset['Amount_in_USD'])





