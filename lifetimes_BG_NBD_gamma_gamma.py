#!/usr/bin/env python
# coding: utf-8

# In[283]:


from lifetimes.datasets import load_cdnow_summary
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from matplotlib import pyplot as plt
from lifetimes.plotting import plot_period_transactions

from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
from lifetimes import GammaGammaFitter

from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data

from lifetimes.utils import calibration_and_holdout_data
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

from lifetimes.plotting import plot_history_alive

from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value


# # Basic Frequency/Recency analysis using the BG/NBD model

# In[284]:


data = load_cdnow_summary(index_col=[0])


# In[285]:


print(data.head())


# In[286]:


# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])


# In[287]:


print(bgf)


# In[288]:


print(bgf.summary)


# ## Visualizing our Frequency/Recency Matrix

# In[289]:


plot_frequency_recency_matrix(bgf)


# In[290]:


plot_probability_alive_matrix(bgf)


# ## Ranking customers from best to worst

# In[291]:


t = 1
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, data['frequency'], data['recency'], data['T'])
data.sort_values(by='predicted_purchases').tail(5)


# ## Assessing model fit

# In[292]:


plot_period_transactions(bgf)


# # Example using transactional datasets

# In[293]:


transaction_data = load_transaction_data()


# In[294]:


print(transaction_data.head())


# In[295]:


summary = summary_data_from_transaction_data(transaction_data, 'id', 'date', observation_period_end='2014-12-31')


# In[296]:


print(summary.head())


# In[297]:


bgf.fit(summary['frequency'], summary['recency'], summary['T'])


# ## More model fitting

# In[298]:


summary_cal_holdout = calibration_and_holdout_data(transaction_data, 'id', 'date', calibration_period_end='2014-09-01', observation_period_end='2014-12-31' )


# In[299]:


print(summary_cal_holdout.head())


# In[300]:


bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)


# In[302]:


individual


# # Customer Predictions

# In[301]:


t = 10 #predict purchases in 10 periods
individual = summary.iloc[20]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])


# In[ ]:


##Customer Probability Histories


# In[303]:


id = 35
days_since_birth = 200
sp_trans = transaction_data.loc[transaction_data['id'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'date')


# # Estimating customer lifetime value using the Gamma-Gamma model

# In[304]:


summary_with_money_value = load_cdnow_summary_data_with_monetary_value()
summary_with_money_value.head()
returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency']>0]


# In[305]:


print(returning_customers_summary.head())


# # The Gamma-Gamma model and the independence assumption

# In[306]:


returning_customers_summary[['monetary_value', 'frequency']].corr()


# In[307]:


ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])


# In[308]:


print(ggf)


# In[309]:


print(ggf.conditional_expected_average_profit(summary_with_money_value['frequency'], summary_with_money_value['monetary_value']).head(10))


# In[310]:


print("Expected conditional average profit: %s, Average profit: %s" % (
    ggf.conditional_expected_average_profit(
        summary_with_money_value['frequency'],
        summary_with_money_value['monetary_value']
    ).mean(),
    summary_with_money_value[summary_with_money_value['frequency']>0]['monetary_value'].mean()
))


# In[311]:


# refit the BG model to the summary_with_money_value dataset
bgf.fit(summary_with_money_value['frequency'], summary_with_money_value['recency'], summary_with_money_value['T'])


# In[312]:


print(ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    summary_with_money_value['frequency'],
    summary_with_money_value['recency'],
    summary_with_money_value['T'],
    summary_with_money_value['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
).head(10))


# # Saving and loading model

# ## Fit model

# In[313]:


from lifetimes import BetaGeoFitter
from lifetimes.datasets import load_cdnow_summary

data = load_cdnow_summary(index_col=[0])
bgf = BetaGeoFitter()
bgf.fit(data['frequency'], data['recency'], data['T'])
bgf
"""<lifetimes.BetaGeoFitter: fitted with 2357 subjects, a: 0.79, alpha: 4.41, b: 2.43, r: 0.24>"""


# ## Saving model

# In[314]:


bgf.save_model('bgf.pkl')
# or
bgf.save_model('bgf_small_size.pkl', save_data=False, save_generate_data_method=False)


# ## Loading model

# In[315]:


bgf_loaded = BetaGeoFitter()
bgf_loaded.load_model('bgf.pkl')
bgf_loaded
"""<lifetimes.BetaGeoFitter: fitted with 2357 subjects, a: 0.79, alpha: 4.41, b: 2.43, r: 0.24>"""


# In[ ]:


bgf_loaded


# In[ ]:




