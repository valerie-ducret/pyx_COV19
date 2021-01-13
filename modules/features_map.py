# This was taken from the tutorial created by Jason Brownlee

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras import layers
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision, AUC
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model():
    model = keras.models.load_model('../model/fine_tuned_vgg16_second_model.h5', custom_objects = {'f1_m' : f1_m})
    return model 

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def select_and_features_map(image, n_layer):
    # load the model
    model = get_model()
    # redefine model to output right after the first hidden layer
    model_select = Model(inputs=model.inputs, outputs=model.layers[n_layer].output)
    # load the image with the required shape
    img = load_img(image, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    features_map = model_select.predict(img)
    # We know the result will be a feature map with 224x224x64. We can plot all 64 two-dimensional images as an 8×8 square of images.
    # or plot a sublist of 64 maps in an 8x8 squares
    square = 8
    ix = 1
    fig = plt.figure()
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            ax.imshow(features_map[0, :, :, ix-1], aspect='auto', cmap='viridis')
            ix += 1
    # show the figure
    st.pyplot(fig)