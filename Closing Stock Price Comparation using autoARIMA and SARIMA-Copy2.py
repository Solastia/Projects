#!/usr/bin/env python
# coding: utf-8

# # Perbandingan SARIMA dan autoARIMA pada Forecast Closing Price Saham Telkom
# 
# # Andara Najla Jilan (10060220025) - EWAKO

# In[1]:


#import library yang diperlukan


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[ ]:


#import dataset
df = pd.read_csv("TLKM.JK.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df["Date"] = pd.to_datetime(df["Date"])


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#missing value check
df.isna().sum()


# In[ ]:


#cek duplikasi data
df.duplicated().sum()


# In[ ]:


#missing value treatment
df.dropna(inplace=True)


# In[ ]:


#hasil treatment missing value
df.isna().sum()


# In[ ]:


df["Date"] = pd.to_datetime(df["Date"])


# In[ ]:


#date sebagai index
df = df.set_index("Date")


# In[ ]:


df


# # Data Visualization 

# In[ ]:


fig = make_subplots(rows=6, cols=1, 
                    subplot_titles=("Opening Price", "Closing Price", "Highest Price", 
                                    "Lowest Price", "Adjusted Closing Price", "Volume"))

fig.add_trace(go.Scatter(x=df.index, y=df["Open"]), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"]), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["High"]), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Low"]), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"]), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Volume"]), row=6, col=1)
fig.update_layout(showlegend=False, height=1200, width=800)
fig.show()


# Data dipisah berdasarkan "Opening Price", "Closing Price", "Highest Price", "Lowest Price", "Adjusted Closing Price", dan "Volume"

# In[ ]:


fig = go.Figure(data=go.Ohlc(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]))
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(title_text="OHLC Chart", title_x=0.5)
fig.show()


# In[18]:


simple_ma = df["Close"].rolling(window=100).mean()

plt.figure(figsize=(14,8))
simple_ma.plot(label="Simple Moving Average")
df["Close"].plot(label="Closing Price")
plt.xticks(rotation=0)
plt.title("Moving Average of Closing Price", size=15)
plt.legend()
plt.show()


# Sampai sini kita bisa lihat data seasonalnya bersifat monthly(bulanan)

# In[19]:


#seasonal decomposition

results = seasonal_decompose(df["Close"], model="multiplicative", period=252)
fig = results.plot()
fig.set_size_inches(12, 8)
fig.tight_layout()
plt.show()


# In[20]:


#data split
df = df.resample("MS").sum()


# In[21]:


df 


# In[22]:


df = df.reset_index()


# In[23]:



size = int(len(df)*0.8)
train = df.loc[:size,["Date", "Close"]]
test = df.loc[size+1:, ["Date", "Close"]]


# train dan test data pada closing price

# In[24]:


df = df.set_index("Date")


# In[25]:


df


# In[26]:


print("Train size:", len(train))
print("Test size:", len(test))
print("Is sum of train and test sizes equal to whole data size:", len(train)+len(test) == df["Close"].shape[0])


# In[27]:


train = train.set_index("Date")
test = test.set_index("Date")


# In[28]:


def adf_test(data):
    result = adfuller(data)
    print(f'ADF Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Number of Lags: {result[2]}')
    print(f'Number of Observations Used: {result[3]}')
    print('Critial Values:')
    for key, value in result[4].items():
        print(f'\t{key}, {value}')


# In[29]:


#uji stasioneritas data
adf_test(train["Close"])


# Data sudah stasioner

# In[30]:


differenced_df = train["Close"] - train["Close"].shift()


# In[31]:


#menghilangkan stasioneritas data
adf_test(differenced_df.dropna())


# In[32]:


plt.figure(figsize=(12, 6))
differenced_df.plot()
plt.title("First Difference of Closing Price")
plt.show()


# In[33]:


#menentukan value p dan q

fig, ax = plt.subplots(2 ,1, figsize=(10, 12))
plot_acf(differenced_df.iloc[1:], lags=20, ax=ax[0])
plot_pacf(differenced_df.iloc[1:], lags=20, ax=ax[1])
plt.show()


# # Model Building

# In[34]:


#untuk melihat prediksi dari 1 September 2022
df.drop("2022-09-01", axis=0, inplace=True)


# In[35]:


df


# In[36]:


df.index


# In[37]:


df = df.reset_index()


# In[38]:


sarima = sm.tsa.statespace.SARIMAX(df.loc[:48, "Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
results = sarima.fit()


# In[39]:


results.summary()


# # Proses Forecasting

# In[40]:


df["forecast_train"] = results.predict(start=0, end=73)


# In[41]:


df[["Close", "forecast_train"]].plot(figsize=(13,6))
plt.title("Forecasting on Train Set", size=25)
plt.show()


# closing price dan forecast_train memiliki nilai yang berdekatan dengan sedikit fluktuasi

# In[42]:


df["forecast_test"] = results.forecast(11)


# In[43]:


df["forecast_test"]


# In[44]:


df.tail(20)


# Data yang tidak NaN adalah data forecast_test

# In[45]:


df[["Close", "forecast_test"]].plot(figsize=(13,6))
plt.title("Forecasting on Test Set", size=15)
plt.show()


# In[46]:


print("Model Performance on Train Set")
print("-"*20)
print("Mean Absolute Error (MAE):", mean_absolute_error(df.loc[:48,"Close"], df.loc[:48,"forecast_train"]))
print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(df.loc[:48,"Close"], df.loc[:48,"forecast_train"]))


# In[47]:


print("Model Performance on Test Set")
print("-"*30)
print("Mean Absolute Error (MAE):", mean_absolute_error(df.loc[49:,"Close"], df.loc[49:,"forecast_test"]))
print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(df.loc[49:,"Close"], df.loc[49:,"forecast_test"]))


# In[48]:


df


# In[49]:


df = df.set_index("Date")


# In[50]:


df


# In[51]:


#Menentukan future forecast

future_dates=[df.index[-1]+ pd.DateOffset(months=x) for x in range(1,13)]
future_df = pd.DataFrame(index=future_dates, columns=df.columns)
df_for_forecast = pd.concat([df, future_df])

df_for_forecast = df_for_forecast.reset_index()
df_for_forecast.rename(columns={"index": "Date"}, inplace=True)


# In[52]:


df_for_forecast["future_forecast"] = results.predict(start=60, end=71)


# In[53]:


df_for_forecast["future_forecast"].tail(15)


# In[54]:


df_for_forecast = df_for_forecast.set_index("Date")


# In[55]:


df_for_forecast["future_forecast"].tail(13)


# Di atas merupakan data yang akan digunakan untuk forecasting

# In[56]:


df_for_forecast[["Close", "future_forecast"]].plot(figsize=(13,6))
plt.title("Forecasting Closing Price From August 2022 to 2023", size=15)
plt.show()


# Pada plot di atas, terlihat bahwa tren closing price untuk dari Agustus 2022-Agustus 2023 cenderung turun

# # Prediksi closing price menggunakan autoARIMA

# In[57]:


get_ipython().system('pip install pmdarima')


# In[58]:


import pmdarima as pm
from pmdarima.arima.utils import ndiffs

d_val = ndiffs(df['Close'], test='adf')
print('Arima D-value:', d_val)


# In[59]:


Ntest = 12
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]
train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

#Define auto-arima to find best model
model = pm.auto_arima(train['Close'],
                      d = d_val,
                      start_p = 0,
                      max_p = 10,
                      start_q = 0,
                      max_q = 10,
                      stepwise=False,
                      max_order=12,
                      trace=True)


# In[60]:


model.get_params()


# In[61]:


def plot_result(model, data, col_name, Ntest):
    
    params = model.get_params()
    d = params['order'][1]
    
    #in sample prediction
    train_pred = model.predict_in_sample(start=d, end=-1)
    #out of sample prediction
    test_pred, conf = model.predict(n_periods=Ntest, return_conf_int=True)
    
    #plotting real values, fitted values, and prediction values
    fig, ax= plt.subplots(figsize=(15,8))
    ax.plot(data[col_name].index, data[col_name], label='Actual Values')
    ax.plot(train.index[d:], train_pred, color='green', label='Fitted Values')
    ax.plot(test.index, test_pred, label='Forecast Values')
    ax.fill_between(test.index, conf[:,0], conf[:,1], color='red', alpha=0.3)
    ax.legend()
    
    #Evaluasi model dengan RMSE dan MAE
    y_true = test[col_name].values
    rmse = np.sqrt(mean_squared_error(y_true,test_pred))
    mae = mean_absolute_error(y_true,test_pred)

    return rmse, mae


# In[62]:


rmse , mae = plot_result(model, df, 'Close', Ntest=12)
print('Root Mean Squared Error: ', rmse)
print('Mean Absolute Error: ', mae)


# # Kesimpulan

# 1. Data seasonal berupa monthly data
# 2. Pada model SARIMA diperoleh
# - Mean Absolute Error (MAE): 10308.596767870167
# - Mean Absolute Percentage Error (MAPE): 0.14341 (14%)
# 3. Pada autoARIMA diperoleh
# - Root Mean Squared Error:  20601.88012713643
# - Mean Absolute Error:  18492.12765957447
# 4. Train dan Test pada data belum optimal dilakukan karena range dalam dataset yang digunakan kecil.
# 5. Performa SARIMA menunjukkan hasil yang lebih baik dibandingkan autoARIMA berdasarkan plot forecasting.
# 6. Berdasarkan nilai MAE dari kedua model, SARIMA merupakan model yang lebih baik dibanding autoARIMA. 
# 
# 
