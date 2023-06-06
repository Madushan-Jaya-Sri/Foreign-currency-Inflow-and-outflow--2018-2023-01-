import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =============================================================================
# # # # # # # # # # # # # # EDA
# =============================================================================
# =============================================================================


full_inflow_dataset = pd.read_csv("E:\\HNB\\camp\\Datasets-4\\final_inflow_dataset.csv")
full_outflow_dataset = pd.read_csv("E:\\HNB\\camp\\Datasets-4\\final_outflow_dataset.csv")
del full_inflow_dataset['Unnamed: 0']
del full_outflow_dataset['Unnamed: 0']

time_series_inflow_dataset = pd.read_csv("E:\\HNB\\camp\\Datasets-4\\time_series_inflow_dataset.csv")
time_series_outflow_dataset = pd.read_csv("E:\\HNB\\camp\\Datasets-4\\time_series_outflow_dataset.csv")

# =============================================================================
# checking the data
# =============================================================================

full_outflow_dataset[['CIF_ID','TRAN_DATE','TRAN_AMT','CURRDESC','BASLE_CODE_DESC']]

print(full_inflow_dataset.head())
print(full_outflow_dataset.head())

print(full_inflow_dataset.describe())
print(full_outflow_dataset.describe())

print(full_inflow_dataset.info())
print(full_outflow_dataset.info())


full_inflow_dataset['TRAN_DATE']= pd.to_datetime(full_inflow_dataset['TRAN_DATE'])
full_outflow_dataset['TRAN_DATE']= pd.to_datetime(full_outflow_dataset['TRAN_DATE'])

full_inflow_dataset['TRAN_DATE'].max()
full_inflow_dataset['TRAN_DATE'].min()

full_outflow_dataset['TRAN_DATE'].min()
full_outflow_dataset['TRAN_DATE'].max()

full_inflow_dataset.isna().sum()
full_inflow_dataset['BASLE_CODE_DESC'].fillna('Blanks', inplace =True)
full_inflow_dataset['BASLE_CODE_DESC'].unique()


full_outflow_dataset.isna().sum()
full_outflow_dataset['BASLE_CODE_DESC'].fillna('Blanks', inplace =True)
full_outflow_dataset['BASLE_CODE_DESC'].unique()

# =============================================================================
# =============================================================================
# =============================================================================
# # # EDA - Inflow
# =============================================================================
# =============================================================================
# =============================================================================
#%matplotlib qt
# =============================================================================
# univariate analysis
# =============================================================================

# =============================================================================
# histogram for all types of currencies considering the USD converted amount
# =============================================================================

full_outflow_dataset.columns

for i in full_outflow_dataset['CURRDESC'].unique():

#     plt.boxplot(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['TRAN_AMT'])
#     plt.show()



    plt.title("Histogram of "+i+" for past 4 years")
    
    sns.distplot(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['Amount_in_USD'], hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})
    plt.xlabel("Amount in USD")
    plt.axvline(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['Amount_in_USD'].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['Amount_in_USD'].max(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['Amount_in_USD'].min(), color='r', linestyle='dashed', linewidth=1)

    plt.show() 


'''
since all the distributions almost skewed log transformations can be used

'''
for i in full_outflow_dataset['CURRDESC'].unique():
    
    scaled_amounts = np.log(full_outflow_dataset[full_outflow_dataset['CURRDESC']==i]['Amount_in_USD']) 

    plt.title("Histogram of "+i+" for past 4 years")
    
    sns.distplot(scaled_amounts, hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})
    plt.xlabel(i+" Amount")
    plt.axvline(scaled_amounts.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(scaled_amounts.max(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(scaled_amounts.min(), color='r', linestyle='dashed', linewidth=1)

    plt.show()
   
# =============================================================================
# overall histogram- initial
# =============================================================================
   
sns.distplot(full_outflow_dataset['Amount_in_USD'], hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})

plt.title("Histogram of inflow amount in USD for past 4 years")
plt.xlabel("Amount in USD")
plt.axvline(full_outflow_dataset['Amount_in_USD'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(full_outflow_dataset['Amount_in_USD'].max(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(full_outflow_dataset['Amount_in_USD'].min(), color='r', linestyle='dashed', linewidth=1)

plt.show() 



# =============================================================================
# overall histogram- scaled
# =============================================================================

scaled_amounts = np.log(full_outflow_dataset['Amount_in_USD']) 
inf_values = scaled_amounts.isin([np.inf, -np.inf])
scaled_amounts = scaled_amounts[~inf_values]

plt.title("Histogram of transformed inflow amount in USD for past 4 years")

sns.distplot(scaled_amounts, hist=True, kde=True, 
              color = 'darkblue', 
              hist_kws={'edgecolor':'black'},
              kde_kws={'linewidth': 1})
plt.xlabel("Amount in USD")
plt.axvline(scaled_amounts.mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(scaled_amounts.max(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(scaled_amounts.min(), color='r', linestyle='dashed', linewidth=1)

plt.show()


# =============================================================================
# from scipy.stats import kstest
# 
# stat, p = kstest(np.log(full_outflow_dataset['Amount_in_USD']), 'norm')
# 
# if p > 0.05:
#     print("Normal")
# else:
#     print("not normal")
# =============================================================================



# =============================================================================
# scatterplot
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

full_outflow_dataset['TRAN_DATE']= pd.to_datetime(full_outflow_dataset['TRAN_DATE'])

plt.scatter(full_outflow_dataset['TRAN_DATE'],full_outflow_dataset['Amount_in_USD'])  
plt.title('Scatter plot of USD amount')
plt.ylabel("Amount in USD")
plt.xlabel("Time")  
plt.show()
#scatter plot with clusters- basles
 
groups = full_outflow_dataset.groupby('BASLE_CODE_DESC')

# Plot

fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.TRAN_DATE ,group.Amount_in_USD, marker='o', linestyle='', ms=5, label=name)

plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
plt.title('Scatter plot by Basles')
plt.ylabel("Amount in USD")
plt.xlabel("Time")
plt.show()



# =============================================================================
# boxplots  
# =============================================================================
plt.title("Boxplot for transformed inflow amount in USD for past 4 years")    
sns.boxplot(y='CURRDESC', x = np.log(full_outflow_dataset['Amount_in_USD'])  , data = full_outflow_dataset)
plt.xlabel("log(Amount in USD)")
full_outflow_dataset[['Amount_in_USD','CURRDESC']]

# =============================================================================
# correlation matrix
# =============================================================================

full_outflow_dataset.columns

corr_matrix = full_outflow_dataset[['TRAN_AMT','NOSTRO_AMT','REF_AMT','MID_RATE','REV_RATE','usd_rate', 'Tran_in_LKR', 'Amount_in_USD',
       'AVG(T.FDINT)']].corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr_matrix, annot=True )
plt.title('Inflow Correlation Matrix')
plt.show()

# =============================================================================
# bar plot of currency types
# =============================================================================
counts = full_outflow_dataset['CURRDESC'].value_counts().sort_values()                               

sns.countplot(y='CURRDESC', data = full_outflow_dataset , order = counts.index).set(title = 'Transactions by Currency types',ylabel = 'Currency Type', xlabel ='No. of Transactions')
plt.show()

# =============================================================================
# bar plot of basle types - counts 
# =============================================================================
counts = full_outflow_dataset['BASLE_CODE_DESC'].value_counts().sort_values()                               

sns.countplot(y='BASLE_CODE_DESC', data = full_outflow_dataset , order = counts.index)
ax = plt.gca()
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'   {int(width)}', (x + width - 0.1, y + height / 2),
                ha='left', va='center', fontsize=12, fontstyle='italic')



plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.title('Transactions by Basle types', fontweight = 'bold', fontsize = 15)  
plt.ylabel( 'Basle Type',fontweight='bold', fontsize=12)
plt.xlabel( 'No. of Transactions', fontweight='bold', fontsize=12)
plt.show()

# =============================================================================
# bar plot of basle types - amounts 
# =============================================================================
amounts = full_outflow_dataset[['Amount_in_USD','BASLE_CODE_DESC']].groupby(by='BASLE_CODE_DESC').sum()['Amount_in_USD'].reset_index().sort_values(by= 'Amount_in_USD')                          

sns.barplot(x='Amount_in_USD', data = amounts ,y='BASLE_CODE_DESC' )


ax = plt.gca()
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'  {int(width)}', (x + width + 0.1, y + height / 2),
                ha='left', va='center', fontsize=12)
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.title('Transaction Amount by Basle types', fontweight = 'bold', fontsize = 15)  
plt.ylabel( 'Basle Type',fontweight='bold', fontsize=12)
plt.xlabel( 'Transaction Amount (USD)', fontweight='bold', fontsize=12)

plt.show()



# =============================================================================
# bar plot for currency types - average amount in USD
# =============================================================================

Avg_amounts = full_outflow_dataset[['Amount_in_USD','CURRDESC']].groupby(by='CURRDESC').mean()['Amount_in_USD'].reset_index().sort_values(by= 'Amount_in_USD')                          

sns.barplot(x='Amount_in_USD', data = Avg_amounts ,y='CURRDESC')


ax = plt.gca()
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'  {int(width)}', (x + width + 0.1, y + height / 2),
                ha='left', va='center', fontsize=12)
plt.xticks(fontsize=11,fontweight='bold')
plt.yticks(fontsize=11,fontweight='bold')
plt.title('Average Transaction Amount by Currency types', fontweight = 'bold', fontsize = 15)  
plt.ylabel( 'Currency Type',fontweight='bold', fontsize=12)
plt.xlabel( 'Avg.Transaction Amount (USD)', fontweight='bold', fontsize=12)

plt.show()

# =============================================================================
# stacked bar chart of currency types by year by avg.amount in USD
# =============================================================================

Avg_amounts_y = full_outflow_dataset[['T_year_y','Amount_in_USD','CURRDESC']].groupby(by=['T_year_y','CURRDESC']).mean()['Amount_in_USD'].reset_index().sort_values(by= 'T_year_y')                          
Avg_amounts_y = Avg_amounts_y[Avg_amounts_y['CURRDESC']!='LKR']

# pivot the dataframe to get the total amount for each currency in each year
df_pivot = Avg_amounts_y.pivot_table(index='T_year_y', columns='CURRDESC', values='Amount_in_USD', aggfunc='mean')

# create stacked bar chart
df_pivot.plot(kind='bar', stacked=True)

# set labels and title
plt.xlabel('Year')
plt.ylabel('Amount')
plt.title('Total Amount by Currency and Year')

# show the plot
plt.show()


# =============================================================================
# dividing into two categories as retail and coparates.
# =============================================================================

retail = (full_outflow_dataset['BASLE_CODE_DESC']=='Blanks') | (full_outflow_dataset['BASLE_CODE_DESC']=='UNCLASSIFIED')|(full_outflow_dataset['BASLE_CODE_DESC']=='INDIVIDUALS')
Retail_dataset_outflow = full_outflow_dataset[retail]

Coperative_dataset_outflow = full_outflow_dataset[~retail]

DF_grouped_R =Retail_dataset_outflow.groupby(by= 'TRAN_DATE').agg({'Amount_in_USD':'sum','usd_rate':'mean','M_inf_rate':'mean','AVG(T.FDINT)':'mean'})
DF_grouped_R.isna().sum()

DF_grouped_C =Coperative_dataset_outflow.groupby(by= 'TRAN_DATE').agg({'Amount_in_USD':'sum','usd_rate':'mean','M_inf_rate':'mean','AVG(T.FDINT)':'mean'})
DF_grouped_C.isna().sum()
# =============================================================================
# creating dates manually
# =============================================================================



dates= pd.date_range('01-01-2018','12-31-2022').to_list()
dates = pd.DataFrame(index=dates)
dates.reset_index(inplace=True)
dates['TRAN_DATE']=dates['index']
del dates['index']
time_series_outflow_retail_dataset = pd.merge(dates,DF_grouped_R, how = 'left',on = 'TRAN_DATE')
time_series_outflow_coperative_dataset = pd.merge(dates,DF_grouped_C, how = 'left',on = 'TRAN_DATE')

time_series_outflow_retail_dataset.isna().sum()
time_series_outflow_coperative_dataset.isna().sum() 
# =============================================================================
# filling null records
# =============================================================================

time_series_outflow_retail_dataset.isna().sum()
time_series_outflow_retail_dataset.fillna(method='ffill',inplace=True)
time_series_outflow_retail_dataset.fillna(method='bfill',inplace=True)

time_series_outflow_retail_dataset.dtypes

time_series_outflow_retail_dataset.index= time_series_outflow_retail_dataset['TRAN_DATE']
del time_series_outflow_retail_dataset['TRAN_DATE']


time_series_outflow_coperative_dataset.isna().sum()
time_series_outflow_coperative_dataset.fillna(method='ffill',inplace=True)
time_series_outflow_coperative_dataset.fillna(method='bfill',inplace=True)

time_series_outflow_coperative_dataset.dtypes

time_series_outflow_coperative_dataset.index= time_series_outflow_coperative_dataset['TRAN_DATE']
del time_series_outflow_coperative_dataset['TRAN_DATE']

# =============================================================================
# saving the datasets
# =============================================================================

time_series_outflow_retail_dataset.to_csv('time_series_outflow_retail_dataset.csv')
time_series_outflow_coperative_dataset.to_csv('time_series_outflow_coperative_dataset.csv')

Retail_dataset_outflow.to_csv('final_retail_outflow_dataset.csv')
Coperative_dataset_outflow.to_csv('final_coperative_outflow_dataset.csv')

#%matplotlib qt

# =============================================================================
# EDA for timeseries
# =============================================================================

time_series_outflow_dataset.columns
time_series_outflow_dataset['TRAN_DATE'] = pd.to_datetime(time_series_outflow_dataset['TRAN_DATE'])

sns.lineplot(y=time_series_outflow_dataset['Amount_in_USD'], x =time_series_outflow_dataset['TRAN_DATE'])
sns.lineplot(y=time_series_outflow_dataset['usd_rate'], x =time_series_outflow_dataset['TRAN_DATE'])
sns.lineplot(y=time_series_outflow_dataset['M_inf_rate'], x =time_series_outflow_dataset['TRAN_DATE'])
sns.lineplot(y=time_series_outflow_dataset['AVG(T.FDINT)'], x =time_series_outflow_dataset['TRAN_DATE'])

# =============================================================================
# histograms
# =============================================================================


for i in time_series_outflow_dataset.drop(['TRAN_DATE'],axis = 1).columns:

#     plt.boxplot(full_inflow_dataset[full_inflow_dataset['CURRDESC']==i]['TRAN_AMT'])
#     plt.show()



    plt.title("Histogram of "+i+" for past 4 years")
    
    sns.distplot(time_series_outflow_dataset[i], hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})
    plt.xlabel(i)
    plt.axvline(time_series_outflow_dataset[i].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(time_series_outflow_dataset[i].max(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(time_series_outflow_dataset[i].min(), color='r', linestyle='dashed', linewidth=1)

    plt.show() 




# =============================================================================
# ACF and PACF plot
# =============================================================================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plot_acf(time_series_outflow_dataset['Amount_in_USD'], lags = 50 )
plot_pacf(time_series_outflow_dataset['Amount_in_USD'], lags = 50)

plt.show()

# =============================================================================
# pairplots
# =============================================================================

sns.pairplot(time_series_outflow_dataset,diag_kind = 'hist')
plt.show()

'''
plots are skewed
'''

# # # EDA - Outflow
