#!/usr/bin/env python
# coding: utf-8

# # Import Library dan Load Data

# Pergerakan harga saham HCL Technologies. HCL Technologies adalah perusahaan konsultan dan layanan teknologi informasi multinasional India yang berkantor pusat di Noida. Ini adalah anak perusahaan dari HCL Enterprise
# 
# - Open adalah harga pembukaan saham pada saat transaksi dimulai dalam suatu periode transaksi. Biasanya, harga pembukaan sama dengan harga penutupan transaksi hari sebelumnya.
# 
# - High menunjukkan harga tertinggi yang pernah terjadi dalam suatu periode perdagangan.
# 
# - Low menunjukkan harga terendah yang pernah terjadi dalam suatu periode perdagangan.
# 
# - Close menunjukkan harga penutupan suatu saham dalam satu hari perdagangan.
# 
# - Adj. Close adalah harga penutupan saham yang sudah disesuaikan (adjusted closing price)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 6 
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[2]:


data = pd.read_csv("HCLTECH.csv")


# In[3]:


data


# # Eksplorasi Data

# In[4]:


type(data)


# In[5]:


data.isnull().sum()/len(data)


# In[6]:


data = data.drop(['Open','Low','Close','Adj Close','Volume'],axis='columns')


# In[7]:


print(data.dtypes)


# In[8]:


data


# In[9]:


#Mengubah tanggal menjadi index
con = data['Date']
data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
#check datatype of index
data.index


# In[10]:


data


# In[11]:


#Menginisialkan variabel High sebagai data time series
ts = data['High']
ts.head(10)


# In[12]:


#Visualisasi dataset
plt.figure(figsize=(20,10))
sns.set_style('darkgrid')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.title('HCL Stock Market High Price')
plt.plot(data['High'])


# ## Uji Stasioneritas Data

# In[13]:


#Deklarasi fungsi untuk mengecek stasioneritas data
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# Uji Hipotesis:
# 
# H0 : Data tidak stasioner
# 
# Ha : Data stasioner
# 
# Tolak H0 apabila diperoleh p-value < alpha (0,05)

# In[14]:


#Melakukan uji stasioneritas untuk dataset
test_stationarity(ts)


# Diperoleh nilai p-value > 0,05, maka H0 tidak ditolak.
# 
# Jadi, data memiliki keadaan yang tidak stasioner.

# Karena data belum stasioner, sehingga perlu dilakukan proses stasionerisasi data, yakni dengan proses differencing untuk memperoleh data yang stasioner.

# In[15]:


ts_diff_1 = ts - ts.shift()
ts_diff_1 = ts_diff_1.dropna()
plt.plot(ts_diff_1)


# In[16]:


test_stationarity(ts_diff_1)


# Setelah dilakukan proses differencing dan diuji stasioneritas data (setelah differencing), diperoleh data setelah differencing 1x telah membentuk data yang stasioner.

# In[17]:


plt.figure()
plt.subplot(211)
plot_acf(ts_diff_1, ax=plt.gca(), lags=100)
plt.subplot(212)
plot_pacf(ts_diff_1, ax=plt.gca(), lags=100)
plt.show()


# Plot ACF dan plot PACF dibentuk untuk mengetahui orde dari AR dan MA, sehingga bisa diketahui model yang sesuai untuk data.

# In[18]:


def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.7)               #data trainingnya 70%
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[19]:


p_values = range(0, 1)
d_values = range(0, 1)
q_values = range(0, 1)
warnings.filterwarnings("ignore")
evaluate_models(ts.values, p_values, d_values, q_values)


# Setelah dilakukan percobaan dengan model ARIMA(p,d,q) dengan:
# 
# - orde p = 0 dan 1
# - orde d = 0 dan 1
# - orde q = 0 dan 1
# 
# Diperoleh model yang menghasilkan RMSE terkecil adalah model ARIMA (1,1,1).
# 
# Hal ini sesuai dengan plot ACF dan PACF yang dies down setelah lag ke-1 dan dataset telah mengalami 1x proses differencing.

# In[20]:


#Visualisasi perbandingan antara dataset setelah differencing dengan data hasil prediksi ARIMA(1,1,1)
model = sm.tsa.arima.ARIMA(ts, order=(1,1,1))
results = model.fit()
predictions_ARIMA = pd.Series(results.fittedvalues,copy=True)
plt.plot(ts_diff_1)             
plt.plot(predictions_ARIMA, color='red')  


# In[22]:


#Informasi yang diperoleh dari pemodelan ARIMA(1,1,1) pada dataset
results.summary()


# # Prediksi Data

# In[ ]:


#Prediksi data dengan model ARIMA(1,1,1) untuk 4 hari ke depan
predict_dif = results_ARIMA.predict(start=1328,end=1331)
predictions_dif_cum_sum = predict_dif.cumsum()                   #prediksi differencingnya
pred_ts = [ts[-1]]
for i, j in enumerate(predictions_dif_cum_sum):                  #prediksi data aslinya
  a = pred_ts[i] + j
  pred_ts.append(a)
predict = pd.Series(pred_ts[1:], index=predict_dif.index)
print(predict)


# Diperoleh hasil prediksi untuk nilai harga tertinggi saham HCLTECH pada periode tanggal :
# - 03 September 2022 = 939,81..
# - 04 September 2022 = 940,73..
# - 05 September 2022 = 944,22..
# - 06 September 2022 = 946,04..

# In[23]:


## KESMIPULAN ##
1. modifikasi pada orde p,d,q dengan value (0,1)
2. Penggunaan tsaplots pada ARIMA
3. Hasilnya yaitu perbedaan pada values dan plot yang dihasilkan oleh ARIMA

