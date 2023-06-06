import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =============================================================================
# # # # # # # # # # # # # # EDA
# =============================================================================

full_inflow_dataset = pd.read_csv("E:\\HNB\\camp\\Datasets-4\\final_inflow_dataset.csv")

del full_inflow_dataset['Unnamed: 0']


time_series_inflow_dataset = pd.read_csv('E:\\HNB\\camp\\Datasets-4\\final two datasets\\TS_inflow.csv')


# =============================================================================
# checking the data
# =============================================================================

full_inflow_dataset[['CIF_ID','TRAN_DATE','TRAN_AMT','CURRDESC','BASLE_CODE_DESC']]

print(full_inflow_dataset.head())
print(full_inflow_dataset.describe())
print(full_inflow_dataset.info())

full_inflow_dataset['TRAN_DATE']= pd.to_datetime(full_inflow_dataset['TRAN_DATE'])

full_inflow_dataset['TRAN_DATE'].max()
full_inflow_dataset['TRAN_DATE'].min()

full_inflow_dataset.isna().sum()
full_inflow_dataset['BASLE_CODE_DESC'].fillna('Blanks', inplace =True)
full_inflow_dataset['BASLE_CODE_DESC'].unique()

# =============================================================================
# =============================================================================
# # # EDA - Inflow
# =============================================================================
# =============================================================================
# =============================================================================
# overall histogram- initial
# =============================================================================
   
sns.distplot(full_inflow_dataset['Amount_in_USD'], hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})
plt.title("Histogram of inflow amount in USD for past 4 years")
plt.xlabel("Amount in USD")
plt.axvline(full_inflow_dataset['Amount_in_USD'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(full_inflow_dataset['Amount_in_USD'].max(), color='r', linestyle='dashed', linewidth=1)
plt.axvline(full_inflow_dataset['Amount_in_USD'].min(), color='r', linestyle='dashed', linewidth=1)

plt.show() 
# =============================================================================
# overall histogram- scaled
# =============================================================================

scaled_amounts = np.log(full_inflow_dataset['Amount_in_USD']) 

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
# bar plot of currency types
# =============================================================================
counts = full_inflow_dataset['CURRDESC'].value_counts().sort_values()                               

sns.countplot(y='CURRDESC', data = full_inflow_dataset , 
              order = counts.index).set(title = 'Transactions by Currency types',
                                        ylabel = 'Currency Type', xlabel ='No. of Transactions')
plt.show()
# =============================================================================
# bar plot of basle types - counts 
# =============================================================================
counts = full_inflow_dataset['BASLE_CODE_DESC'].value_counts().sort_values()                               

sns.countplot(y='BASLE_CODE_DESC', data = full_inflow_dataset , order = counts.index)
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
# pie charts for retail and coperate - counts
# =============================================================================

full_inflow_dataset['r_c']=np.where((full_inflow_dataset['BASLE_CODE_DESC']== 'INDIVIDUALS') | 
                                    (full_inflow_dataset['BASLE_CODE_DESC']== 'STAFF') | 
                                    (full_inflow_dataset['BASLE_CODE_DESC']== 'UNCLASSIFIED') | 
                                    (full_inflow_dataset['BASLE_CODE_DESC']== 'Blanks'), 
                                    'Retail' , 'Co-operate')
df_pie = full_inflow_dataset['r_c'].value_counts().reset_index()
fig, ax = plt.subplots()
ax.pie(df_pie['r_c'], labels=df_pie['index'], autopct='%1.1f%%', startangle=90, 
       pctdistance=0.8, wedgeprops=dict(width=0.4))

# Add circle to create a hole
circle = plt.Circle((0, 0), 0.4, color='white')
fig.gca().add_artist(circle)

# Set plot title
ax.set_title("Distribution of Sales")
# Show the plot
plt.show()

# =============================================================================
# bar plot of basle types - amounts 
# =============================================================================
amounts = full_inflow_dataset[['Amount_in_USD','BASLE_CODE_DESC']].groupby(
    by='BASLE_CODE_DESC').sum()['Amount_in_USD'].reset_index().sort_values(by= 'Amount_in_USD')                          

sns.barplot(x='Amount_in_USD', data = amounts ,y='BASLE_CODE_DESC' )

ax = plt.gca()
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'  {int(width)}', (x + width + 0.1, y + height / 2),
                ha='left', va='center', fontsize=9)
plt.xticks(fontsize=9,fontweight='bold')
plt.yticks(fontsize=9,fontweight='bold')
plt.title('Transaction Amount by Basle types', fontweight = 'bold', fontsize = 15)  
plt.ylabel( 'Basle Type',fontweight='bold', fontsize=12)
plt.xlabel( 'Transaction Amount (USD)', fontweight='bold', fontsize=12)

plt.show()

# =============================================================================
# bar plot for currency types - average amount in USD
# =============================================================================

Avg_amounts = full_inflow_dataset[['Amount_in_USD','CURRDESC']].groupby(
    by='CURRDESC').mean()['Amount_in_USD'].reset_index().sort_values(by= 'Amount_in_USD')                          

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

Avg_amounts_y = full_inflow_dataset[['T_year_y','Amount_in_USD','CURRDESC']].groupby(
    by=['T_year_y','CURRDESC']).mean()['Amount_in_USD'].reset_index().sort_values(by= 'T_year_y')                          
Avg_amounts_y = Avg_amounts_y[Avg_amounts_y['CURRDESC']!='LKR']

# pivot the dataframe to get the total amount for each currency in each year
df_pivot = Avg_amounts_y.pivot_table(index='T_year_y', columns='CURRDESC',
                                     values='Amount_in_USD', aggfunc='mean')

# create stacked bar chart
df_pivot.plot(kind='bar', stacked=True)
# set labels and title
plt.xlabel('Year')
plt.ylabel('Amount')
plt.title('Total Amount by Currency and Year')

# show the plot
plt.show()




# =============================================================================
# EDA for timeseries
# =============================================================================

time_series_inflow_dataset.columns
time_series_inflow_dataset['TRAN_DATE'] = pd.to_datetime(time_series_inflow_dataset['TRAN_DATE'])

sns.lineplot(y=time_series_inflow_dataset['Amount_in_USD'], x =time_series_inflow_dataset['TRAN_DATE'])


sns.lineplot(y=time_series_inflow_dataset['usd_rate'], x =time_series_inflow_dataset['TRAN_DATE'])
sns.lineplot(y=time_series_inflow_dataset['M_inf_rate'], x =time_series_inflow_dataset['TRAN_DATE'])
sns.lineplot(y=time_series_inflow_dataset['AVG(T.FDINT)'], x =time_series_inflow_dataset['TRAN_DATE'])

# =============================================================================
# histograms
# =============================================================================


for i in time_series_inflow_dataset.drop(['TRAN_DATE'],axis = 1).columns:
    plt.title("Histogram of "+i+" for past 4 years")
    
    sns.distplot(time_series_inflow_dataset[i], hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 1})
    plt.xlabel(i)
    plt.axvline(time_series_inflow_dataset[i].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(time_series_inflow_dataset[i].max(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(time_series_inflow_dataset[i].min(), color='r', linestyle='dashed', linewidth=1)

    plt.show() 

# =============================================================================
# correlation matrix
# =============================================================================

time_series_inflow_dataset.columns

corr_matrix = time_series_inflow_dataset.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr_matrix, annot=True )
plt.title('Inflow Correlation Matrix')
plt.show()

# removing usd_rate
corr_matx = time_series_inflow_dataset.drop("usd_rate", axis =1).corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr_matx, annot=True )
plt.title('Inflow Correlation Matrix')
plt.show()
# =============================================================================
# pairplots
# =============================================================================

sns.pairplot(time_series_inflow_dataset.iloc[:,:5],diag_kind = 'hist')
plt.show()


