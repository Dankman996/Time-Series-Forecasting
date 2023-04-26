#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas_datareader as web 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix


# In[2]:


pip install pandas_datareader


# In[5]:


pip install pandas


# In[16]:


pip install seaborn


# In[6]:


pip install yfinance


# In[7]:


import plotly.graph_objs as go
import yfinance as yf


# In[10]:


stock=input("enter a stock")
print(stock)


# In[11]:


df = yf.download(tickers=stock,period='1d',interval='1m')
df1=df.head(100)
print(df1)
data=pd.DataFrame(df)
#print the data we have requested
print(data)


# In[12]:


df1= yf.download(tickers=stock,period='1d',interval='1m')
print(df1)


# # **DATA PREPARATION**

# In[13]:


wiki=pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Investing_in_the_NASDAQ-100')
df=pd.DataFrame(wiki[4])
print(df1)
df1=df[df.columns[0:2]]
l1=list(df1['Ticker'])
l2=list(df1['Company'])
print(l2)
len(l1)


# In[14]:


data=pd.DataFrame()
print(data)


# In[15]:


for i in l1[0:2]:
  df = yf.download(tickers=i,period='1d',interval='1m')
  print(df)


# In[16]:


k=0
for i in l1[0:2]:
  df = yf.download(tickers=i,period='1d',interval='1m')
  print(df.shape[0])
  df.insert(0,'Stock Symbol',i,True)
  df.insert(1, 'Company',l2[k], True)
  if(k!=len(l2)):
    k=k+1
  else:
    continue
  data=pd.concat([data,df],ignore_index=False)
  print(data)


# In[17]:


k=0
for i in l1:
  df = yf.download(tickers=i,period='1d',interval='1m')
  print(df.shape[0])
  df.insert(0,'Stock Symbol',i,True)
  df.insert(1, 'Company',l2[k], True)
  if(k!=len(l2)):
    k=k+1
  else:
    continue
  data=pd.concat([data,df],ignore_index=False)
  print(data)


# In[19]:


data.to_csv("Top 100 stocks.csv")


# # **Final data code- Method 1**

# In[20]:


wiki=pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Investing_in_the_NASDAQ-100')
df=pd.DataFrame(wiki[4])
# print(df1)
df1=df[df.columns[0:2]]
l1=list(df1['Ticker'])
l2=list(df1['Company'])
# print(l2)
# len(l1)
data=pd.DataFrame()
print(data)
k=0
for i in l1:
  df = yf.download(tickers=i,period='1d',interval='1m')
  print(df.shape[0])
  df.insert(0,'Stock Symbol',i,True)
  df.insert(1, 'Company',l2[k], True)
  if(k!=len(l2)):
    k=k+1
  else:
    continue
  data=pd.concat([data,df],ignore_index=False)
  print(data)

data.to_csv("Top 100 stocks.csv")


# In[25]:


data1=data


# In[22]:


data1.head()


# # **Data Preprocessing**

# In[26]:


fin_data=data
fin_data.head(10)


# # **Importing the dataset- Method 2**

# In[27]:


df=pd.read_csv("Top 100 stocks.csv")
df.head()
df['Stock Symbol'].unique()


# ## **shape of data**

# In[28]:


df.columns


# In[29]:


df.shape


# # **statistics**

# In[30]:


df.describe()


# # **Checking Null values**

# In[31]:


df.isnull().sum()


# # **Visualization & Correlation**

# In[32]:


for i in df.columns[3:]:
  print(i)


# In[33]:


for i in df.columns[3:]:
  print(i)
  print(df.hist(column=i, bins=50))


# In[34]:


df.iloc[:, 3:8].plot(kind='hist',
        alpha=0.7,
        bins=25,
        title='Histogram Of stocks',
        rot=45,
        grid=True,
        figsize=(15,8),
        fontsize=15
        )


# In[35]:


df[df['Stock Symbol']=='TSLA']['Close'].plot(figsize=(16,8))


# In[36]:


df['Close'].plot(figsize=(16,8))


# In[37]:


df_new=df
df_new=df_new.reset_index()

df_new=df_new[['Datetime','Close','Stock Symbol']]
df_new.head()
df_new_pivot=df_new.pivot('Datetime','Stock Symbol', 'Close').reset_index()
df_new_pivot.head()


# In[38]:


corr_df = df_new_pivot.corr(method='pearson')
corr_df.head().reset_index()
corr_df.head(10)


# In[39]:


plt.figure(figsize=(50, 8))
sns.heatmap(corr_df.head(10), annot=True)
plt.figure()


# line plot of 'CLOSE' for the Apple company.

# In[40]:


df1.columns


# In[41]:


while(True):
  process=str(input('can we proceed (yes/no) ?'))
  if(process=='yes'):
    print('Choose from list of stocks: ')
    print(df_new['Stock Symbol'].unique())
    stock=(input('Enter the stock symbol: '))
    stock=stock.upper()
    df1= yf.download(tickers=stock,period='1d',interval='1m')
    for i in df1.columns:
      print(df1.plot.line(y=i,use_index=True))
      print("---------- plot for "+ i + " -----------")
  else:
    print("No problem ! ")
    break


# # **Moving Averages**

# Since the stock prices changed very quickly, we can find the moving averages to find how the price of each stock gonna change.

# In[44]:


df2=df
df2


# In[45]:


unq_list=df2['Stock Symbol'].unique()
while(True):
  process=str(input('can we proceed (yes/no) ?'))
  if(process=='yes'):
    print('Choose from list of stocks: ')
    print(unq_list)
    stock=(input('Enter the stock symbol: '))
    stock=stock.upper()
    df_ind= yf.download(tickers=stock,period='1d',interval='1m')
    df_ind['MA 50']=df_ind['Adj Close'].rolling(50).mean()
    df_ind['MA 100']=df_ind['Adj Close'].rolling(100).mean()
    print(df_ind[['Adj Close','MA 50','MA 100']].plot(figsize=(16,8)))
  else:
    print("No problem ! ")
    break


# In both of these plots above, we can see how the price of the stock changes from second to second on the date 04/11.

# In[48]:


df_ind


# # **Target variable declared**

# In[47]:


df_ind=df_ind.dropna(axis='columns')


# In[ ]:


df_ind.head()


# In[49]:


data=df2[['Close']]
data=data.rename(columns = {'Close':'Actual_Close'})
data


# In[ ]:


data["Target"] = df2.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]


# In[ ]:


data['datetime']=df2['Datetime']
data.head(10)


# In[ ]:


df2.head(10)


# In[ ]:


df2_past = df2.copy()
df2_prev = df2_past.shift(1)
df2_prev.head(2)


# In[ ]:


predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(df2_prev[predictors]).iloc[1:]


# In[ ]:


data.head()


# # **Shuffle data**

# In[ ]:


data1=data.reindex(np.random.permutation(data.index))
data1.head(10)


# # **Training a machine learning model**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
model = RandomForestClassifier(n_estimators=100, min_samples_split=300, random_state=1)


# # **Splitting the data**

# In[ ]:


train = data1.iloc[:-100]
test = data1.iloc[-100:]


# In[ ]:


train.shape


# In[ ]:


test.shape


# # **Training the model with train data**

# In[ ]:


model.fit(train[predictors], train["Target"])


# # **Measuring Error**

# In[ ]:


# Evaluate error of predictions
pred = model.predict(test[predictors])
pred = pd.Series(pred, index=test.index)
precision_score(test["Target"], pred)


# Random forest classifier is predicting the future price of the data based on the past data with 60% accuracy.

# # **Plotting the input target and predict target**

# In[ ]:


output=pd.DataFrame(pred)
output['test output']=test['Target']
output.columns=['Prediction','Test Output']
output.head(10)


# In[43]:


output.plot()


# # **DM Models**

# # **Logistic regression**

# In[ ]:


data1.head()


# In[ ]:


train_lr=data1.iloc[:,[0,3,4,5,6,7]]
test_lr=data1.iloc[:, [1]]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_lr,test_lr,test_size=0.3, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)*100))


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
cm


# # **Support Vector Machines**

# In[ ]:


model = svm.SVC(kernel='poly', degree=2)
model.fit(X_train, y_train)


# In[ ]:


predictions_poly = model.predict(X_test)
accuracy_poly = accuracy_score(y_test, predictions_poly)
print("2nd degree polynomial Kernel\nAccuracy (normalized): " + str(accuracy_poly))


# # **Stock price of Tesla after Musk took over the twitter**

# In[ ]:


tesla= yf.download(tickers="TSLA",period='10d',interval='1d')
print(tesla)


# In[ ]:


tesla.plot.line(figsize=(16,8))


# In[ ]:


tesla['Adj Close'].plot(figsize=(16,8))


# When we observe the trend of the tesla Stock, I found out that the musk take over the twitter on 11-27-2022. since the news got out on the night itself, many investors prebooked the tesla stocks. As we can observe from the graph, the price reached its high on 28 october 2022. From the next 3 days, the price undergoes little change in the price of the stock. Recently a debate is going on about musk changing rules in twitter, this makes the price goes down to  $205. The same variation with the twitter too.
