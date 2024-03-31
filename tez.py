# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from keras import utils
from keras import layers
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing import image_dataset_from_directory



def myModel(train_dir, test_dir, model_path='myModel.keras'):
  # Veri setini yükleme ve ön işleme
  train_dataset =image_dataset_from_directory('C:/Users/berna/Downloads/CiltHastaliklari/train/train',
                                             shuffle=True,
                                            # bir seferde işlenecek görüntü sayısı
                                             batch_size=32,
                                             image_size=(256, 256))
  test_dataset = image_dataset_from_directory('C:/Users/berna/Downloads/CiltHastaliklari/test/test',
                                            shuffle=True,
                                            batch_size=32,
                                            image_size=(256, 256))
  # Modeli oluşturma
  num_classes = 4
  ##num_classes değeri, modelin tahmin etmesi gereken sınıf sayısını belirtir.
  ##Bu, modelin çıkış katmanındaki nöron sayısını ayarlamak için kullanılır ve
  ##modelin çıktısının, veri setindeki farklı sınıfları temsil edecek şekilde boyutlandırılmasını sağlar
  model = Sequential([
  layers.Rescaling(1./255, input_shape=(256, 256, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
  ])
  # Modeli derleme
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  ##sparse_categorical_crossentropy=modelin tahmin ettiği olasılıklar ile gerçek
  ##sınıf etiketleri arasındaki farkı hesaplar ve modelin bu farkı azaltacak şekilde eğitilmesini sağlar
  # Modeli eğitme
  model.fit(train_dataset, validation_data=test_dataset, epochs=10)
  ##epochs==kaç kere işleneceği
  ##bu sayı çok olursa ezberleme olur

  loss, accuracy = model.evaluate(test_dataset)
  print(f"Test kaybı: {loss}, Test doğruluğu: {accuracy}")


  #modeli kaydetme
  model.save(model_path)
  print(f"Model {model_path} olarak kaydedildi.")
  return model


# myModel fonksiyonunu çağır ve modeli al
model = myModel('C:/Users/berna/Downloads/CiltHastaliklari/train/train', 'C:/Users/berna/Downloads/CiltHastaliklari/test/test')

