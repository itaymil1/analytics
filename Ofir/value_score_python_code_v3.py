import numpy as np
import pandas as pd
from datetime import date, timedelta
from matplotlib import pyplot as plt
import matplotlib.style as style
from matplotlib.pylab import rcParams
import seaborn as snsa
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from google.cloud import bigquery
import warnings
warnings.filterwarnings('ignore')

# BigQuery connection
def query(query, project_name = 'rapyd-bq-poc-2020'):
    client = bigquery.Client(project = project_name)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df

# K-means 
def k_means(df, column, num_clusters = 5, quan = .99, name = ''):

    if quan is not None:
        threshold = df[column].quantile(quan)
        new = df[df[column] < threshold].copy()
        outliers = df[df[column] >= threshold].copy()
    else:
        new = dfx.copy()

    kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++', random_state = 42)
    values_reshape = new.loc[:,column].values.reshape(-1,1)
    y_kmeans = kmeans.fit_predict(values_reshape)

    values = new.loc[:,column].copy()
    temp = pd.DataFrame({'y_kmeans':y_kmeans, 'values':values})
    
    borders = []
    for i in range(num_clusters):
        min_border = temp[temp.y_kmeans == i]['values'].min()
        borders = np.append(borders, min_border)
    borders.sort()

    new_col = name
    new[new_col] = None
    for i, v in enumerate(borders, start = 1):
        new[new_col] = np.where(new[column] >= v, i, new[new_col])
        
    if quan is not None:
        outliers[new_col] = num_clusters
        new = pd.concat([new,outliers], ignore_index = True)

    # print(new.groupby(new_col)[column].agg(['count','min','max']))     

    return new
    
# Cross Join function 
def fill_missing_weeks(df, time_col = 'trx_created_week', 
                       obj_cols = ['first_trx','last_trx','total_merchant_trx','merchant_name','signup_country','mcc','kyb_status','rapyd_entity_name'],
                       num_cols = ['num_transactions','trx_amount_usd']):

    a = df.copy()
    
    a[time_col] = pd.to_datetime(a[time_col])
    start, end = a[time_col].agg(['min','max']) 

    weeks = pd.DataFrame(pd.date_range(start = start, end = end, freq = 'W-MON'), columns = ['week'])

    cross = pd.MultiIndex.from_product([a.merchant_id.unique(), weeks.week.unique()], 
                                       names = ['merchant_id', time_col]).to_frame(index = False)

    b = cross.merge(a, on = ['merchant_id', time_col], how = 'left')
    
    for col in obj_cols:
        b[col] = b.groupby('merchant_id')[col].transform(lambda x: x.ffill().bfill())
        
    for col in num_cols:
        b[col] = pd.to_numeric(b[col]).fillna(0).round(2)
    
    b = b[b.trx_created_week >= b.first_trx].copy()
    b.sort_values(by = ['merchant_id',time_col], ignore_index = True, inplace = True)

    return b

# Calculate trends per merchant
def calc_trends(df, smoothing_level = 0.2, col = None, prefix = None):
    
    df.set_index('trx_created_week', inplace = True)
    df[col] = df[col].astype(int)

    # Exponential smoothing
    result = SimpleExpSmoothing(df[col]).fit(smoothing_level = smoothing_level)
    predicted = result.predict(start = 0, end = len(df)-1)
    next_week = result.forecast(steps = 1)
    last_exp_value = predicted.iloc[-1:].values[0]
    predicted_value = next_week.values[0]
    change = (predicted_value / last_exp_value - 1)
    
    # Linear regression
    X = pd.Series(range(len(df))).values.reshape(-1,1)
    y = df[col].values.reshape(-1,1)
    linear = LinearRegression().fit(X,y)
    lin_trend = linear.coef_[0][0].round(2)
    lin_predict = linear.predict([[X[-1][0]+1]])
    
    return pd.DataFrame({'merchant_id':[df['merchant_id'].iloc[0]], 
                         'merchant_name':[df['merchant_name'].iloc[0]],
                         'median_w_amount':[df['median_w_amount'].iloc[0]], 
                         'active_weeks':[df['active_weeks'].iloc[0]],
                         'num_transactions':[df['num_trx_last'].iloc[0]],
                         'trx_amount_usd':[df['trx_amount_last'].iloc[0]],
                         f'{prefix}_exp_trend':[change], 
                         f'{prefix}_exp_pred':[predicted_value], 
                         f'{prefix}_lin_trend':[lin_trend],
                         f'{prefix}_lin_pred':([lin_predict[0][0]])})

# Time series plot
def time_series_plot(df, merchant, col):
    
    plt.rcParams.update({'figure.figsize':(16,5), 'font.size':10})
    
    t1 = df[df.merchant_id == merchant][['trx_created_week',col]].copy()
    
    t2 = t1.copy()
    t2.set_index('trx_created_week', inplace = True)
    t2[col] = t2[col].astype(int)

    # A higher smoothing level assigns greater importance to recent observations
    smoothing_level = 0.3

    result = SimpleExpSmoothing(t2[col]).fit(smoothing_level = smoothing_level)
    predicted = result.predict(start = 0, end = len(t2)-1)
    next_week = result.forecast(steps = 1)

    last_exp_value  = predicted.iloc[-1:].values[0]
    predicted_value = next_week.values[0]
    
    X = t1.index.values.reshape(-1,1)
    y = t1[col].values.reshape(-1,1)
    linear = LinearRegression().fit(X,y)
    lin_pred = linear.predict(X)
    lin_next_week = linear.predict([[X[-1][0] +1]])

    #### Import data
lookback_window = 365 

data = query(f"""
select date(timestamp_trunc(transaction_timestamp,week(monday))) as trx_created_week
, transaction_merchant_id as merchant_id
, action_merchant_alias as merchant_name
, action_merchant_signup_country_name as signup_country
, transaction_merchant_mcc_code_description as mcc
, transaction_merchant_type_name as merchant_type
, transaction_merchant_kyb_status as kyb_status
, transaction_merchant_rapyd_entity as rapyd_entity_name
, count(1) as num_transactions
, sum(transaction_direction_amount_usd) as trx_amount_usd
from rapyd-data.rapyd_master.master_fact_rapyd_transactions
where transaction_action_type='payment' 
and (NOT action_merchant_is_test  OR action_merchant_is_test IS NULL)
and not transaction_is_merchant_moment
and transaction_timestamp>= TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -{lookback_window} DAY)
and transaction_timestamp< (TIMESTAMP_ADD(TIMESTAMP_ADD(TIMESTAMP_TRUNC(TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY), WEEK(MONDAY)), INTERVAL (-1 * 7) DAY), INTERVAL (1 * 7) DAY))
group by 1,2,3,4,5,6,7,8
""")

data['trx_amount_usd'] = pd.to_numeric(data['trx_amount_usd'])

df = data.copy()

# 1st & last transaction
df['first_trx'] = df.groupby('merchant_id').trx_created_week.transform('min')
df['last_trx'] = df.groupby('merchant_id').trx_created_week.transform('max')

df['first_trx'] = pd.to_datetime(df['first_trx'])
df['last_trx'] = pd.to_datetime(df['last_trx'])

# total transactions
df['total_merchant_trx'] = df.groupby('merchant_id').num_transactions.transform('sum')

# fill missing weeks
df = fill_missing_weeks(df)
df.sort_values(by = ['merchant_id','trx_created_week'], ignore_index = True, inplace = True)

# avg & median weekly amount
df['avg_w_amount'] = df.groupby('merchant_id').trx_amount_usd.transform('mean')
df['median_w_amount'] = df.groupby('merchant_id').trx_amount_usd.transform('median')

# last week values
df['num_trx_last'] = df.groupby('merchant_id').num_transactions.transform('last')
df['trx_amount_last'] = df.groupby('merchant_id').trx_amount_usd.transform('last')

#### Keep relevant population

df['since_last_trx'] = (pd.to_datetime(date.today()) - df['last_trx']).dt.days

#### Keep only recent data (last 13 weeks)

last_13w = np.sort(df.trx_created_week.unique())[-13:]

df = df[df.trx_created_week.isin(last_13w)].sort_values(by = ['merchant_id','trx_created_week'], ignore_index = True).copy()

# Threshold for active week
activity_threshold = df[df.num_transactions > 0].num_transactions.quantile(.05)

active_weeks = df[df.num_transactions > activity_threshold].groupby('merchant_id', as_index = False).agg(active_weeks = ('trx_created_week','count'))
df = df.merge(active_weeks, on = 'merchant_id', how = 'left')
df['active_weeks'] = df['active_weeks'].fillna(0).astype(int)

# conditions
condition_1 = (df.total_merchant_trx >= 500)
condition_2 = (df.first_trx <= pd.to_datetime(date.today()) - timedelta(days = 150))
condition_3 = (df.since_last_trx <= 60)
condition_4 = condition_4 = (df.active_weeks > 7)
   
df = df[condition_1 & condition_2 & condition_3 & condition_4].reset_index(drop = True).copy()


#### Calculate trends per merchant
num = df.groupby('merchant_id').\
      apply(lambda group: calc_trends(group, col = 'num_transactions', prefix = 'num')).\
      reset_index(drop = True)

usd = df.groupby('merchant_id').\
      apply(lambda group: calc_trends(group, col = 'trx_amount_usd', prefix = 'usd')).\
      reset_index(drop = True)

# merged dataframe
on_cols = ['merchant_id', 'merchant_name', 'median_w_amount', 
           'active_weeks', 'num_transactions', 'trx_amount_usd']
new = num.merge(usd, on = on_cols, how = 'inner')

# clip to zero
clip_columns = [col for col in new.columns if 'pred' in col]
new[clip_columns] = new[clip_columns].clip(lower = 0)
new = new.round(2)

#### Division to clusters

# Size groups
new = k_means(df = new, column = 'median_w_amount', num_clusters = 3, name = 'size_group', quan = .9)

di1 = {1:'1.Small', 2:'2.Medium', 3:'3.Large'}
new.replace({'size_group':di1}, inplace = True)

# Trend columns (positive or negative)
trend_cols = ['num_exp_trend','num_lin_trend','usd_exp_trend','usd_lin_trend']
new['positive_cols'] = new[trend_cols].applymap(lambda x: 1 if x > 0 else 0).sum(axis = 1).fillna(0).astype(int)

# Trend cluster
new = k_means(df = new, column = 'num_exp_trend', num_clusters = 5, name = 'trend_group', quan = .99)

di2 = {1:'1.Negative trend', 
       2:'2.Stable trend', 
       3:'2.Stable trend', 
       4:'2.Stable trend',
       5:'3.Positive trend'}
new.replace({'trend_group':di2}, inplace = True)

# Modifications
new.loc[(new.trend_group == '3.Positive trend') & 
        ((new.positive_cols < 4)), 'trend_group'] = '2.Stable trend'
new.loc[(new.trend_group == '1.Negative trend') & (new.positive_cols > 0), 'trend_group'] = '2.Stable trend'

new['week'] = data.trx_created_week.max()
col = new.pop('week')
new.insert(0, 'week', col)


#### Append to BigQuery
final = new.copy()

final['trend_group'] = final['trend_group'].str[2:].copy()
final['size_group'] = final['size_group'].str[2:].copy()

project_id = 'rapyd-bq-poc-2020'
dataset_id = 'gcf_main'
table_id   = 'value_score_payments'
full_table_id = f'{project_id}.{dataset_id}.{table_id}'

# delete potential duplications
#query(f"""delete from {project_id}.{dataset_id}.{table_id} where week = '{final.week.max()}'""")

# append new records
#final.to_gbq(destination_table = full_table_id, project_id = project_id, if_exists = 'append')