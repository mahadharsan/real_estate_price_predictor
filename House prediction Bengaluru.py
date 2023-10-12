#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mp 


# In[2]:


df = pd.read_csv("Bengaluru_House_Data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.groupby('area_type').agg('count')


# In[6]:


df2 = df.drop(['availability','society','balcony','area_type'], axis = 'columns')


# In[7]:


df2.head()


# In[8]:


df2.isnull().sum()


# In[9]:


df3 = df2.dropna()


# In[10]:


df3.isnull().sum()


# In[11]:


df3['size'].unique()


# In[12]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[13]:


df3.head()


# In[14]:


df3['bhk'].unique()


# In[15]:


df3[df3.bhk>20]


# In[16]:


df3.total_sqft.unique()


# In[17]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[18]:


df3[~df3.total_sqft.apply(is_float)]


# In[19]:


def convert_to_avg(x):
    nums = x.split("-")
    if len(nums)==2:
        return (float(nums[0])+ float(nums[1]))/2
    try:
        return float(x)
    except:
        return None


# In[20]:


df3.total_sqft.apply(convert_to_avg)


# In[21]:


convert_to_avg('2660 No')


# In[22]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_to_avg)
df4.head()


# In[48]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# In[49]:


df5['location'].nunique()


# In[50]:


df5['location'] = df5['location'].apply(lambda x: x.strip())


# In[51]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)


# In[52]:


len(location_stats[location_stats <=10])


# In[53]:


location_stats_less_10 = location_stats[location_stats <=10]  
location_stats_less_10


# In[54]:


len(df5['location'].unique())


# In[55]:


df5['location'].nunique()


# In[56]:


df5['location'] = df5['location'].apply(lambda x: 'other' if x in location_stats_less_10 else x)
df5['location'].nunique()


# In[57]:


df5.head()


# In[58]:


#clearing outliers 
df5[df5['total_sqft']/df5['bhk']<300]


# In[59]:


df6 = df5[~(df5['total_sqft']/df5['bhk']<300)]


# In[60]:


df5.shape


# In[61]:


df6.shape


# In[63]:


df6.price_per_sqft.describe()


# In[64]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st))& (subdf.price_per_sqft<= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape


# In[67]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape
    


# In[71]:


import matplotlib.pyplot as plt

# Assuming you have imported pandas and have a DataFrame called df8

plt.rcParams['figure.figsize'] = (20, 10)
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price per square feet")
plt.ylabel("Count")
plt.show()  # Add this line to display the histogram


# In[73]:


plt.hist(df8.bath,rwidth=0.5)
plt.xlabel('Number of bathrooms')
plt.ylabel('count')


# In[91]:


df8[df8['bath']>(df8['bhk']+2)]


# In[92]:


df9 = df8[df8['bath']<(df8['bhk']+2)]


# In[93]:


df9.shape


# In[95]:


df10 = df9.drop(['size','price_per_sqft'], axis='columns' )
df10.head()


# In[102]:


f = pd.get_dummies(df10['location'])


# In[103]:


df11 = pd.concat([df10,f.drop('other',axis='columns')],axis='columns')


# In[104]:


df11.head()


# In[105]:


df12 = df11.drop('location', axis = 'columns')


# In[106]:


df12.head()


# In[ ]:





# In[107]:


X = df12.drop('price',axis='columns')
X.head()


# In[108]:


Y = df12['price']
Y.head()


# In[109]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[111]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)
lr_clf.score(X_test,Y_test)


# In[112]:


def predict_price (location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]= bhk
    if loc_index >= 0:
        x[loc_index]=1
    return lr_clf.predict([x])[0]


# In[113]:


predict_price('Indira Nagar',1000,2,2)


# In[114]:


import pickle
with open('blr_home_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[115]:


import json 
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))


# In[ ]:




