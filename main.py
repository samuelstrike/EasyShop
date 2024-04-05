import pandas as pd
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-computed features and filenames
feature_list = np.array(pickle.load(open('features_embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Load styles data
styles_df = pd.read_csv('styles.csv', usecols=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName'])
styles_df['id'] = styles_df['id'].astype(str)

# Merge features and filenames with styles data
filenames_df = pd.DataFrame({'id': [os.path.basename(filepath).split('.')[0] for filepath in filenames], 'filepath': filenames})
merged_df = pd.merge(filenames_df, styles_df, on='id', how='left')

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar products based on features
def recommend(features, feature_list, merged_df, k=5):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    recommended_products = merged_df.iloc[indices[0]]
    return recommended_products

# Streamlit app

def display_recommended_products(recommended_products, num_columns):
    num_rows = -(-len(recommended_products) // num_columns)  # Ceiling division
    for i in range(num_rows):
        row_data = recommended_products.iloc[i * num_columns: (i + 1) * num_columns]
        cols = st.columns(num_columns)
        for col, (_, product) in zip(cols, row_data.iterrows()):
            display_product(col, product)

def display_product(column, product):
    product_name = product["masterCategory"]
    column.write(product_name)
    img = Image.open(product['filepath'])
    column.image(img, caption=product["productDisplayName"], width=img.width, use_column_width=False)


st.title('ShopEasy')
option = st.sidebar.radio('Select Option', ('Product Name', 'Upload Image', 'Take Picture'))

if option == 'Product Name':
    product_name = st.sidebar.text_input('Enter product name')

    if product_name:
        
        merged_df_cleaned = merged_df.dropna(subset=['productDisplayName'])
       
        matching_products = merged_df_cleaned[merged_df_cleaned['productDisplayName'].str.contains(product_name, case=False)]
        if not matching_products.empty:
            
            matching_features = np.array([feature_extraction(img_path, model) for img_path in matching_products['filepath']])
            
            features_avg = np.mean(matching_features, axis=0)
            recommended_products = recommend(features_avg, feature_list, merged_df)
            if not recommended_products.empty:
                st.subheader('Recommended Products')
                num_columns = 5
                display_recommended_products(recommended_products, num_columns)

elif option == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.sidebar.image(img, caption='Uploaded Image', use_column_width=False)
        st.image(img, caption='Uploaded Image', use_column_width=False) 
        features = feature_extraction(uploaded_file, model)
        recommended_products = recommend(features, feature_list, merged_df)
        st.subheader('You may also like this')
        num_columns = 5
        display_recommended_products(recommended_products, num_columns)

elif option == 'Take Picture':
    picture = st.camera_input("Take a picture")

    if picture:
        st.image(picture)
        features = feature_extraction(picture, model)
        recommended_products = recommend(features, feature_list, merged_df)
        st.subheader('You may also like this')
        num_columns = 5
        display_recommended_products(recommended_products, num_columns)




