# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle


resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False


model = tf.keras.Sequential([
    resnet_model,
    GlobalMaxPooling2D()
])

print(model.summary())

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


filenames = [os.path.join('images', file) for file in os.listdir('images')]


feature_list = [extract_features(file, model) for file in tqdm(filenames)]


with open('features_embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)
