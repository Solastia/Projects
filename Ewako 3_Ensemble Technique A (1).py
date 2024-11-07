#!/usr/bin/env python
# coding: utf-8

# # Import Library dan Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Dataset status brain stroke berdasarkan variabel-variabel yang mempengaruhi
data = pd.read_csv('brain_stroke.csv')


# # Eksplorasi Data

# In[3]:


#Menampilkan 5 baris teratas dari dataset
data.head()


# In[4]:


#Menampilkan 5 baris terakhir dari dataset
data.tail()


# In[5]:


#Menampilkan ukuran dimensi dari dataset
data.shape


# In[6]:


#Mengetahui adanya data yang duplikat
data.duplicated().sum()


# In[7]:


#Mengetahui data yang bernilai kosong
data.isnull().sum()


# In[8]:


#Mengetahui informasi umum dari dataset
data.info()


# In[9]:


data.describe()


# In[10]:


data.nunique()


# In[11]:


#Mengelompokkan variabel dataset yang berupa kategori
data_cat = data[['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']]


# In[12]:


data_cat


# In[13]:


#Mengetahui nilai kategori dari tiap variabel
for i in data_cat.columns:
  print(data_cat[i].unique())


# In[14]:


#Mengetahui frekuensi tiap kategori dari variabel dataset
for i in data_cat.columns:
  print(data_cat[i].value_counts())


# ## Visualisasi Data

# In[15]:


for i in data_cat.columns:
  plt.figure(figsize = (15,6))
  sns.countplot(data_cat[i],data=data_cat,palette='hls')
  plt.xticks(rotation = 90)
  plt.show()


# Berdasar visualisasi di atas, dapat diperoleh informasi berupa :
# 1. dataset memiliki frekuensi jenis kelamin perempuan yang lebih banyak dibandingkan laki-laki
# 2. dataset memiliki frekuensi penderita hipertensi yang lebih sedikit dibandingkan yang bukan penderita hipertensi
# 3. dataset memiliki frekuensi bukan penderita heart disease yang lebih banyak dibandingkan yang penderita hipertensi
# 4. dataset memiliki frekuensi yang pernah menikah lebih banyak daripada yang belum pernah menikah
# 5. dataset memiliki 4 tipe pekerjaan yang didominasi oleh tipe pekerjaan private
# 6. dataset memiliki penghuni residence dengan tipe urban dan rural yang hampir seimbang
# 7. dataset memiliki frekuensi tidak pernah merokok lebih banyak dibandingkan kategori yang lain
# 8. dataset memiliki lebih banyak yang tidak menderita stroke dibandingkan dengan yang menderita stroke

# In[16]:


for i in data_cat.columns:
  plt.figure(figsize=(15,6))
  data_cat[i].value_counts().plot(kind='pie',autopct='%1.1f%%')
  plt.xticks(rotation=90)
  plt.show()


# In[17]:


for i in data_cat.columns:
  plt.figure(figsize=(15,6))
  sns.countplot(data_cat[i], data = data_cat, hue='stroke', palette='hls')
  plt.xticks(rotation=90)
  plt.show()


# In[18]:


data['ever_married'] = [0 if i !='Yes' else 1 for i in data['ever_married']]


# In[19]:


data['gender'] = [0 if i != 'Female' else 1 for i in data['gender']]


# In[20]:


data.head()


# In[21]:


data = pd.get_dummies(data, columns = ['work_type', 'Residence_type', 'smoking_status'])


# In[22]:


data.sample(10)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[24]:


#Menghapus variabel stroke dari dataset yang masuk sebagai variabel X dan inisialisasi variabel stroke sebagai variabel y
X = data.drop(['stroke'],axis = 1)
y = data['stroke']


# In[25]:


#Membagi dataset menjadi data training dan testing dengan rasio 33% data digunakan sebagai data testing. Dengan pemilihan random data sebesar 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)
X_train.shape, X_test.shape


# # Modelling

# In[26]:


#Membuat model random forest dengan banyak tree dalam forest adalah 10 serta untuk mengukur kualitas split menggunakan entropi untuk perolehan infomasi
classifier_rf = RandomForestClassifier(n_estimators=100, criterion="gini")
classifier_rf.fit(X_train, y_train)


# In[27]:


#Memperoleh nilai y prediksi dari variabel-variabel X data testing
y_pred = classifier_rf.predict(X_test)


# # Evaluation

# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


#Menghitung confusion matrix dari y hasil prediksi dan y nilai asli
cm = confusion_matrix(y_test, y_pred)


# In[30]:


print(cm)


# In[31]:


#Menampilkan nilai akurasi untuk prediksi di data training
print('Training-set accuracy score:', classifier_rf.score(X_train, y_train))


# In[32]:


#Menampilkan nilai akurasi untuk prediksi di data testing
print('Training-set accuracy score:', classifier_rf.score(X_test, y_test))


# In[33]:


#Proporsi kelas 1 (stroke=yes)
y_train.sum()/y_train.count()


# # BALANCING DATA

# Class Imbalance adalah situasi yang terjadi ketika salah satu class memiliki jumlah lebih besar dari pada class lainnya. 
# 
# Paradoks akurasi adalah kondisi dimana akurasi bukanlah metrik yang baik untuk model prediktif ketika mengklasifikasikan class imbalance.
# 
# Salah satu cara penanganannya menggunakan balancing data pada data train. Balancing data bertujuan untuk membuat proporsi kelas mayoritas dan minoritas menjadi seimbang. 
# 
# Salah satu teknik yang populer diterapkan dalam rangka menangani ketidak seimbangan kelas adalah SMOTE (Synthetic Minority Over-sampling Technique). Teknik ini mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan dataset dengan cara sampling ulang sampel kelas minoritas

# In[34]:


pip download imblearn


# In[35]:


pip install imblearn


# In[36]:


from imblearn.over_sampling import SMOTE
# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)


# In[37]:


# MODEL
#Membuat model random forest dengan banyak tree dalam forest adalah 10 serta untuk mengukur kualitas split menggunakan entropi untuk perolehan infomasi
classifier_rf2 = RandomForestClassifier(n_estimators=100, criterion="gini")
classifier_rf2.fit(X_train_SMOTE, y_train_SMOTE)


# In[38]:


#Proporsi kelas 1 (stroke=yes) setelah diSMOTE
y_train_SMOTE.sum()/y_train_SMOTE.count()


# In[39]:


#Memperoleh nilai y prediksi dari variabel-variabel X data testing
y_pred = classifier_rf2.predict(X_test)


# In[40]:


#Menghitung confusion matrix dari y hasil prediksi dan y nilai asli
cm = confusion_matrix(y_test, y_pred)
cm


# In[41]:


#Menampilkan nilai akurasi untuk prediksi di data training
print('Training-set accuracy score:', classifier_rf2.score(X_train_SMOTE, y_train_SMOTE))
#Menampilkan nilai akurasi untuk prediksi di data testing
print('Training-set accuracy score:', classifier_rf2.score(X_test, y_test))


# ## AdaBoost Model

# In[42]:


# --- Applying AdaBoost ---
from sklearn.ensemble import AdaBoostClassifier
ABclassifier = AdaBoostClassifier(n_estimators=100)

ABclassifier.fit(X_train_SMOTE, y_train_SMOTE)
y_pred_AB = ABclassifier.predict(X_test)


# In[43]:


#Menampilkan nilai akurasi untuk prediksi di data training
print('Training-set accuracy score:', ABclassifier.score(X_train_SMOTE, y_train_SMOTE))
#Menampilkan nilai akurasi untuk prediksi di data testing
print('Training-set accuracy score:', ABclassifier.score(X_test, y_test))


# In[45]:


## modifikasi yang dilakukan:
#1. Mengganti test_size dari 0.33 menjadi 0.40
#2. Mengganti n_estimator menjadi 100
#3. Mengganti criterion menjadi gini

## perbedaan yang terjadi:
# 1. Confusion matrix bernilai berbeda
# 2. Training set accuracy menjadi 1.0
# 3. Testing set accuracy nilainya naik sedikit
# 4. Proporsi kelas 


# In[ ]:




