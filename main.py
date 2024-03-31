# main.py

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

def fotograf_yukle_ve_tahmin_et(fotograf_yolu, model):
    img = image.load_img(fotograf_yolu, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model için uygun hale getir

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]

    return predicted_index

if __name__ == '__main__':
    # Modeli yükle
    model = tf.keras.models.load_model('myModel.keras')

    fotograf_yolu = input("Lütfen tahmin için bir görüntü dosya yolu girin: ")
    tahmin = fotograf_yukle_ve_tahmin_et(fotograf_yolu, model)

    class_names = ['Acne and Rosacea', 'Atopic Dermatitis', 'Eczema Photos', 'Psoriasis pictures Lichen Planus and related diseases']

    if tahmin < len(class_names):
        predicted_class = class_names[tahmin]
        print(f"Yapılan tahminin sınıfı: {predicted_class}")
    else:
        print("Tahmin indeksi sınıf isimleri listesinin dışında.")
