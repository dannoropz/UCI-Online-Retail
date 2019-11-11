#!/usr/bin/env python
# coding: utf-8

# I will begin by importing the libraries that will be used and also the data file obtained from the UCI ML repository: 

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# In[2]:


pd.set_option('display.float_format', lambda x: '%.1f' % x)


# In[3]:


online_retail = pd.read_excel(r'C:\Users\Equipo\Documents\Pythonities\Datasets\Online_Retail.xlsx')


# I will now examine the condition of the dataset and see if it's missing some information:

# In[4]:


online_retail.head(5)


# Note that we have the unit price and also the quantity, so we can calculate the purchased amount by multiplying these amounts. Let's do so:

# In[5]:


online_retail['PurchasedAmount'] = online_retail.Quantity*online_retail.UnitPrice


# In[6]:


online_retail.head(5)


# In[7]:


online_retail.info()


# Notice that all columns except CustomerID are >99% complete; CustomerID is missing more than 100,000 values! Let us now see how these missing values are distributed.

# In[8]:


sns.heatmap(online_retail.isnull(), cbar=False)


# So it seems we have missing CustomerIDs everywhere. We can still know where they're from, what they bought, how much, when and for what unit price, though. Let us keep that in mind for now.

# I will now procede to examine each variable to see if we can find anything else that is off-setting. 

# In[9]:


len(online_retail.CustomerID.unique())


# We can see that AT LEAST 4373 customers, from 01/12/2010 to 09/12/2011, bought something from the store. However, we don't know how many more clients the store had because many IDs are missing. We also cannot adequately compute the customers that had more or less purchases based on their IDs, because we can't attribute many of the items to anyone.

# In[10]:


len(online_retail.Country.unique())


# And we know these clients came from 38 different countries.

# In[11]:


len(online_retail.Description.unique())


# If we take a look at the store's item descriptions, they seem to be very idiosyncratic. However, note that, in truth, there are only 4224 different types of items sold. 

# In[12]:


len(online_retail.StockCode.unique())


# In[13]:


online_retail.UnitPrice.describe()


# Given that, numerically, price is a meaningful variable, I want to see how it is distributed: it's maximum, minimum and percentiles. Note that 75% of items sold have a unit price below 4.13. However, also note that the maximum is at 38970. Does this make sense? Moreoever, the minimum is a big negative number! We know a negative price makes no sense, so we definately need to explore this variable further. 

# In[14]:


online_retail.UnitPrice[online_retail.UnitPrice < 0]


# Notice that only two transactions have negative prices. Let see what is their description.

# In[15]:


online_retail.loc[299983:299984,]


# Based on their description, these rows seem to be debt adjustments and not a purchase. 

# But what about rows with a unit price of exactly 0? If we look for them in the dataset, we actually find over 2000! And their description is not very helpful, so we'll remove them too. Notice:

# In[16]:


online_retail[online_retail.UnitPrice == 0]


# Let's now examine prices in the top quartile.

# In[17]:


or_top25 = online_retail[online_retail.UnitPrice > 4.13]


# In[18]:


or_top25.UnitPrice.describe()


# Look how the top 25% of our data still has 75% of its instances under a price of 9. This tells us that there are a couple transactions with high prices, but the great majority of our transactions have a unit price below 10. What are the transactions with the top 50 unit prices? Let's find out:

# In[19]:


online_retail.nlargest(50,'UnitPrice')


# We can immediately notice that most of the big unit price transactions have AMAZON FEE as their description and stock code! They also seem to have a -1 quantity most of the time. There is also 'Manual', which sometimes has a positive and sometimes a negative quantity. These do not seem to be product descriptions; rather, they seem to be other kinds of transactions. 

# Let us see what kind of descriptions we have for instances with a unit price greater than 500 to see if our suspicions are correct!

# In[20]:


or_greater500 = online_retail[online_retail.UnitPrice > 500]


# In[21]:


len(or_greater500)


# We're talking about 255 transactions here.

# In[22]:


or_greater500[['Description','UnitPrice']].groupby('Description').mean()


# Notice how every single one of the descriptions, except PICNIC BASKET, is not a purchase by a customer but rather some other sort of transaction: amazon free, cruk commission, bank charges, etc. We don't really know if these or other similar types of descriptions are spread throughout the dataset, which is problematic for our analysis. We'll assume most of these kinds of transactions are accumulated in the high unit price category, and so by removing high unit prices we can get rid of them. 

# But before removing anything, let's explore the Quantity variable:

# In[23]:


online_retail.Quantity.describe()


# There is something strange here. Like price, Quantity has a large negative number as its minimum (which makes no sense) and a large positive number as its maximum, despite 75% of purchases having a quantity less than 10. We need to explore this just like we did with price. 

# In[24]:


len(online_retail[online_retail.Quantity<0])


# We have a lot (>10000) of transactions with negative quantities (but it's only <2% of our data). Are these returns, mistakes or something else? It is worth exploring, so I'll keep them as a separate dataset. However, right now I want to focus on transactions that result in a positive purchased amount for the store, so I'll drop them from the main dataset.

# In[25]:


or_negative_quant = online_retail[online_retail.Quantity<0]


# In[26]:


or_negative_quant.head()


# Let's now explore the other side of the distribution. Why do some transactions have quantities of 80,000?

# In[27]:


online_retail.nlargest(20,'Quantity')


# In[28]:


len(online_retail[online_retail['Quantity'] >= 1000])


# Except for those records that also have unit price zero or unregistered customer ID, there seems to be nothing wrong here. There could be a typing mistake somewhere, but perhaps some people just buy a lot of things

# So far, we've discovered that we have transactions that do not seem to be related to the purchase of a product with high unit prices. We've also discovered that there are a lot of negative quantity transactions. These two phenomena don't seem to be independent of each other. And finally, we found two transactions with negative unit price. Because this analysis focuses on item purchases for the business, we don't want other types of transactions or negative purchased amounts, so I've decided to drop all rows with a unit price higher than 500 or with a negative quantity, plus all transactions with zero or negative unit prices, like so: 

# In[29]:


online_retail_01 = online_retail[(online_retail.Quantity>0) & (online_retail.Quantity < 1000)& (online_retail.UnitPrice<500) & (online_retail.UnitPrice>0)].copy()


# In[30]:


online_retail_01.info()


# In[31]:


sum(online_retail_01.PurchasedAmount < 0)


# All purchased amounts are now positive. 

# How are these purchased amounts distributed in this clean dataset?

# In[32]:


online_retail_01[['Quantity','UnitPrice','PurchasedAmount']].describe()


# In[33]:


online_retail_01.head()


# In[34]:


online_retail_01[['Country','PurchasedAmount']].groupby('Country').agg({'PurchasedAmount':['sum','mean','median','min','max']}).sort_values(('PurchasedAmount','sum'), ascending=False).round(1).head(5)


# So it seems this UK-based store sells mostly inside the UK and to neighboring European countries. Let's see what are the most frequently sold items in each of these countries.

# In[35]:


data = online_retail_01[online_retail_01.Country == 'United Kingdom'].groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(5)
ax = sns.barplot(x='InvoiceNo', y='Description', data=data.reset_index())
ax.set_title('United Kingdom', fontsize=30)
ax.set_ylabel('Description', fontsize=20)
ax.set_xlabel('Number of purchases', fontsize=20)


# In[36]:


data = online_retail_01[online_retail_01.Country == 'EIRE'].groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(5)
ax = sns.barplot(x='InvoiceNo', y='Description', data=data.reset_index())
ax.set_title('Ireland', fontsize=30)
ax.set_ylabel('Description', fontsize=20)
ax.set_xlabel('Number of purchases', fontsize=20)


# In[37]:


data = online_retail_01[online_retail_01.Country == 'Netherlands'].groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(5)
ax = sns.barplot(x='InvoiceNo', y='Description', data=data.reset_index())
ax.set_title('Netherlands', fontsize=30)
ax.set_ylabel('Description', fontsize=20)
ax.set_xlabel('Number of purchases', fontsize=20)


# In[38]:


data = online_retail_01[online_retail_01.Country == 'Germany'].groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(5)
ax = sns.barplot(x='InvoiceNo', y='Description', data=data.reset_index())
ax.set_title('Germany', fontsize=30)
ax.set_ylabel('Description', fontsize=20)
ax.set_xlabel('Number of purchases', fontsize=20)


# In[39]:


data = online_retail_01[online_retail_01.Country == 'France'].groupby('Description')['InvoiceNo'].count().sort_values(ascending=False).head(5)
ax = sns.barplot(x='InvoiceNo', y='Description', data=data.reset_index())
ax.set_title('France', fontsize=30)
ax.set_ylabel('Description', fontsize=20)
ax.set_xlabel('Number of purchases', fontsize=20)


# In[40]:


online_retail_01.head()


# In[41]:


online_retail_01['InvoiceHour'] = online_retail_01['InvoiceDate'].apply(lambda x: x.hour)


# In[42]:


online_retail_01['InvoiceMonth'] = online_retail_01['InvoiceDate'].apply(lambda x: x.month)


# In[43]:


online_retail_01['InvoiceDateT'] = online_retail_01['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, x.day))


# In[44]:


online_retail_01.head()


# In[45]:


data = online_retail_01.groupby('InvoiceHour')['InvoiceNo'].count()
ax = sns.barplot(x='InvoiceHour',y='InvoiceNo', data=data.reset_index())
ax.set_title('Number of purchases by hour', fontsize=15)
ax.set_xlabel('Hour')
ax.set_ylabel('Number of purchases')


# So we see that most of the store's purchases happen around midday and until 4 PM or so. 

# In[46]:


data = online_retail_01.groupby('InvoiceMonth')['InvoiceNo'].count()
ax = sns.barplot(x='InvoiceMonth',y='InvoiceNo', data=data.reset_index())
ax.set_title('Number of purchases by month', fontsize=15)
ax.set_xlabel('Month')
ax.set_ylabel('Number of purchases')


# We see that, as we approach Christmas and related holidays, the number of purchases at the store increases. They are particularly high in November, as would be expected. 

# ## We will now start with the cohort analysis:

# In[47]:


def get_month(x):
    return dt.datetime(x.year, x.month, 1)


# In[48]:


online_retail_01['PurchaseMonth'] = online_retail_01['InvoiceDate'].apply(get_month)


# In[49]:


online_retail_01['MonthCohort'] = online_retail_01.groupby('CustomerID')['PurchaseMonth'].transform('min')


# In[50]:


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# In[51]:


invoice_year, invoice_month, _ = get_date_int(online_retail_01, 'PurchaseMonth')
cohort_year, cohort_month, _ = get_date_int(online_retail_01, 'MonthCohort')


# In[52]:


years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month


# In[53]:


online_retail_01['CohortIndex'] = years_diff*12 + months_diff


# In[54]:


cohort_data = online_retail_01.groupby(['MonthCohort','CohortIndex'])['CustomerID'].apply(pd.Series.nunique)


# In[55]:


cohort_data = cohort_data.reset_index()


# In[56]:


cohort_counts = cohort_data.pivot(index='MonthCohort',
                                  columns='CohortIndex',
                                  values='CustomerID')


# In[57]:


cohort_counts


# I will now calculate the retention rate:

# In[58]:


cohort_sizes = cohort_counts.iloc[:,0]


# In[59]:


retention = cohort_counts.divide(cohort_sizes, axis=0)


# In[60]:


retention.round(3)*100


# In[76]:


fig = plt.figure(figsize=(15,10))
ax = sns.heatmap(data = retention,
            annot = True,
            fmt = '.0%',
            vmin = 0,
            vmax = .5,
            cmap = 'coolwarm')
ax.set_title('Retention rate by cohort and month', fontsize=20)


# In[62]:


cohort_data = online_retail_01.groupby(['MonthCohort','CohortIndex'])['PurchasedAmount'].mean()


# In[63]:


cohort_data = cohort_data.reset_index()


# In[64]:


cohort_purchases = cohort_data.pivot(index='MonthCohort',
                                     columns='CohortIndex',
                                     values='PurchasedAmount')


# In[75]:


fig = plt.figure(figsize=(15,10))
ax = sns.heatmap(data=cohort_purchases,
            annot = True,
            vmin = 0,
            vmax = 35,
            cmap = 'BuGn')
ax.set_title('Average purchase by Cohort, Month')


# I will now calculate Frequency, Recency and Monetary Value metrics:

# In[66]:


snapshot_date = max(online_retail_01.InvoiceDate) + dt.timedelta(days=1)


# In[67]:


datamart = online_retail_01.groupby('CustomerID').agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                                                       'InvoiceNo':'count',
                                                       'PurchasedAmount':'sum'})


# In[68]:


datamart.rename(columns={'InvoiceDate':'Recency',
                         'InvoiceNo':'Frequency',
                         'PurchasedAmount':'Spend'},inplace=True)


# In[69]:


spend_quartiles = pd.qcut(datamart['Spend'],4,labels=range(1,5))
recency_quartiles = pd.qcut(datamart['Recency'],4,labels=range(4,0,-1))
frequency_quartiles = pd.qcut(datamart['Frequency'],4,labels=range(1,5))


# In[70]:


datamart['M'] = spend_quartiles
datamart['R'] = recency_quartiles
datamart['F'] = frequency_quartiles


# In[71]:


def join_stuff(x): return str(x['M']) + str(x['R']) + str(x['F'])


# In[72]:


datamart['RFM_Segment'] = datamart.apply(join_stuff, axis=1)


# In[73]:


datamart['Score'] = datamart[['M','R','F']].sum(axis=1)


# In[82]:


datamart.head()


# Let's now group customers by their score and analyze their metrics:

# In[80]:


datamart.groupby('Score').agg({'Recency':'mean',
                               'Frequency':'mean',
                               'Spend':['mean','count']}).round()


# ## KMeans Clustering

# In[84]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[87]:


seed = 1


# In[89]:


datamart_rfs = datamart[['Recency','Frequency','Spend']]


# In[90]:


datamart_log = np.log(datamart_rfs)


# In[91]:


scaler = StandardScaler()


# In[92]:


scaler.fit(datamart_log)


# In[93]:


datamart_normalized = scaler.transform(datamart_log)


# In[94]:


sse = {}


# In[95]:


for k in range(1,21):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(datamart_normalized)
    sse[k] = kmeans.inertia_


# In[96]:


ax = sns.pointplot(x=list(sse.keys()),y=list(sse.values()))
ax.set_title('Elbow Method Plot')
ax.set_xlabel('Number of clusters, k')
ax.set_ylabel('SSE')


# In[97]:


kmeans = KMeans(n_clusters=3, random_state=seed)


# In[98]:


kmeans.fit(datamart_normalized)


# In[101]:


cluster_labels = kmeans.labels_


# In[102]:


datamart_rfs_k3 = datamart_rfs.assign(Cluster=cluster_labels)


# In[107]:


cluster_avg = datamart_rfs_k3.groupby('Cluster').mean()


# In[120]:


cluster_avg


# In[106]:


population_avg = datamart_rfs.mean()


# In[111]:


importance = cluster_avg / population_avg - 1


# In[119]:


plt.figure(figsize=(8, 6))
ax = sns.heatmap(importance,
           annot = True,
           cmap='coolwarm')
ax.set_title('RFS relative importance by cluster', fontsize=15)


# In[ ]:




