import os
import numpy as np
from tensorflow import keras

seq_model = keras.models.load_model('app/model/keras_seq_model')
seq_model.metrics_names

vgg16_model = keras.models.load_model('app/model/keras_vgg16_model')
vgg16_model.metrics_names


def translate_pred(prediction: np.array) -> str:
    if prediction[0][0] > 0.5:
        return "Dog", prediction[0][0] * 100
    else:
        return "Cat", 100 - (prediction[0][0] * 100)

def keras_seq_predict(image_uri: str):
    import numpy as np
    from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
    
    #load the image
    img_width, img_height = 150, 150
    my_image = load_img(image_uri, target_size=(img_width, img_height))

    #preprocess the image
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # img_arr = img_to_array(my_image)
    img_arr = np.expand_dims(img_to_array(my_image), axis=0)
    preprocessed_img = next(test_datagen.flow(img_arr, batch_size=1))

    prediction = seq_model.predict(preprocessed_img)
    return translate_pred(prediction)

def keras_vgg16_predict(image_uri: str):
    import numpy as np
    from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
    from keras.applications.vgg16 import preprocess_input

    #load the image
    img_width, img_height = 224, 224
    my_image = load_img(image_uri, target_size=(img_width, img_height))

    #preprocess the image
    img_arr = img_to_array(my_image)
    img_arr = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    my_image = preprocess_input(img_arr)
    prediction = vgg16_model.predict(my_image)
    return translate_pred(prediction)


def identify_animal_kind(file_path, model_variant):
    
    if model_variant == "sequence":
        return keras_seq_predict(file_path)
    else:
        return keras_vgg16_predict(file_path)

def identify_animal_kinds(file_paths):
    data = list()
    for fp in file_paths:
        seq_ak, seq_conf = identify_animal_kind(fp, "sequence")
        vgg_ak, vgg_conf = identify_animal_kind(fp, "vgg16")
        data.append((seq_ak, seq_conf, vgg_ak, vgg_conf))
        
    return data