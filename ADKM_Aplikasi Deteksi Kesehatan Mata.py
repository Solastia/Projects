#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install visualkeras')


# In[ ]:


get_ipython().system('pip install luwiji')


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import cv2
import os
import tensorflow as tf
import pandas as pd
import visualkeras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from torchvision import transforms
from luwiji.cnn import illustration
warnings.filterwarnings('ignore')


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# # Preprocessing (Augmentasi)

# # Pelabelan

# In[ ]:


data = tf.keras.utils.image_dataset_from_directory(directory = '/content/drive/MyDrive/Projek Akhir/Dataset/',
                                                   color_mode = 'rgb',
                                                   batch_size = 16,
                                                   image_size = (224,224),
                                                   shuffle=True,
                                                   seed = 2022)


# In[ ]:


labels = np.concatenate([y for x,y in data], axis=0)


# In[ ]:


values = pd.value_counts(labels)
values = values.sort_index()


# In[ ]:


values


# In[ ]:


class_names = data.class_names
for idx, name in enumerate(class_names):
  print(f"{idx} = {name}", end=", ")


# In[ ]:


plt.figure(figsize=(12,8))
plt.pie(values,autopct='%1.1f%%', explode = [0.02,0.02,0.02,0.02,0.02], textprops = {"fontsize":15})
plt.legend(labels=data.class_names)
plt.show()


# In[ ]:


# os.listdir(path)


# ## Menerapkan data generator untuk eksplorasi data

# In[ ]:


data_iterator = data.as_numpy_iterator()


# In[ ]:


batch = data_iterator.next()


# Setiap kumpulan berisi 64 gambar, setiap gambar berukuran 224x224 *(Berdasar pada ketetapan image size di pelabelan diatas)*

# In[ ]:


batch[0].shape


# ## Menampilkan beberapa data

# In[ ]:


plt.figure(figsize=(10,10))
for images, labels in data.take(1):
    for i in range (9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')


# ## Split data

# In[ ]:


print("Total kumpulan data (bacths) = ",len(data))


# In[ ]:


train_size = int(0.7 * len(data)) +1
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))


# In[ ]:


train_size


# In[ ]:


zval_size


# In[ ]:


test_size


# In[ ]:


train = data.take(train_size)
remaining = data.skip(train_size)
val = remaining.take(val_size)
test = remaining.skip(val_size)


# ### persiapan test set

# In[ ]:


test_iter = test.as_numpy_iterator()


# In[ ]:


test_set = {"images":np.empty((0,224,224,3)), "labels":np.empty(0)}
while True:
    try:
        batch = test_iter.next()
        test_set['images'] = np.concatenate((test_set['images'], batch[0]))
        test_set['labels'] = np.concatenate((test_set['labels'], batch[1]))
    except:
        break


# In[ ]:


y_true = test_set['labels']


# # Awal dari CNN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax


# In[ ]:


# Menampilkan history loss/accuracy
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_his(history):
    plt.figure(figsize=(15,12))
    metrics = ['accuracy', 'loss']
    for i, metric in enumerate(metrics):
        plt.subplot(220+1+i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()


# In[ ]:


def create_baselineCNN():
    model = Sequential([
        Conv2D(filters = 64, kernel_size=3, activation = 'relu', padding='same', input_shape=(224,224,3)),
        Conv2D(filters = 64, kernel_size=3, activation = 'relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),

        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),
        
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        Conv2D(filters = 128, kernel_size=3,padding='same', activation = 'relu',),
        BatchNormalization(),
        MaxPool2D(2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation = 'relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation = 'relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(4, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


# In[ ]:


model = create_baselineCNN()


# In[ ]:


model.summary()

visualkeras.layered_view(model,legend=True)


# # Pelatihan Model
# 
# Kami akan melatih jaringan Nerual dengan 10 epoch dan memberikannya panggilan balik awal jika akurasinya tidak meningkat banyak selama 60 epoch

# In[ ]:


from keras import callbacks 
early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", 
        patience=20,
        verbose=1,
        mode="max",
        restore_best_weights=True,
)

history = model.fit(
    train,
    validation_data=val,
    epochs = 3,
    callbacks=[early_stop],
)


# In[ ]:


plot_his(history)


# ### Evaluasi model dengan test set

# In[ ]:


y_pred = np.argmax(model.predict(test_set['images']), 1)


# In[ ]:


print(classification_report(y_true, y_pred, target_names = class_names))


# In[ ]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(4)+.5, class_names, rotation=90)
plt.yticks(np.arange(4)+.5, class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")



# # Transfer Learning (Pretrained Model)
# 
# Di sini saya akan mencoba menggunakan model pretraind, menyempurnakannya agar sesuai dengan data kami, Anda dapat membaca lebih lanjut tentang pembelajaran transfer di sini
# 
# Saya menggunakan model pra-pelatihan EfficientNet, karena data yang dilatih berbeda dari data kami, saya membuat lapisan tingkat atas dapat dilatih untuk memungkinkannya melatih, menyesuaikan, dan beradaptasi dengan data kami

# In[ ]:


def make_model():
    effnet = EfficientNetB3(include_top=False, weights="imagenet",input_shape=(224,224,3), pooling='max') 
    effnet.trainable=False
    
    for layer in effnet.layers[83:]:
      layer.trainable=True
    
    x = effnet.output
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x = Dropout(rate=.45, seed=2022)(x)        
    output=Dense(4, activation='softmax')(x)
    
    model= tf.keras.Model(inputs=effnet.input, outputs=output)
    model.compile(optimizer = 'adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model


# In[ ]:


model = make_model()


# ## Pelatihan Model

# In[ ]:


from keras import callbacks 
early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", 
        patience=10,
        verbose=1,
        mode="max",
        restore_best_weights=True, 
     )

history = model.fit(
    train,
    validation_data=val,
    epochs = 50,
    callbacks=[early_stop],
)


# In[ ]:


plot_his(history)


# ## Evaluasi Model dengan test set

# In[ ]:


y_pred = np.argmax(model.predict(test_set['images']), 1)


# In[ ]:


print(classification_report(y_true, y_pred, target_names = class_names))


# In[ ]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
plt.xticks(np.arange(4)+.5, class_names, rotation=90)
plt.yticks(np.arange(4)+.5, class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")


