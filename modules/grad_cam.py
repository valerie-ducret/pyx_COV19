import numpy as np
import tensorflow as tf
import keras
import streamlit as st
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.cm as cm
import os.path
import gdown

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
    
    if not os.path.exists("fine_tuned_vgg16_second_model.h5"):
        url = "https://drive.google.com/file/d/13ytKE6ZruB9W-br_31HhxjKGkjGCtrjo/view?usp=sharing"
        output = "fine_tuned_vgg16_second_model.h5"
        gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    model = keras.models.load_model("fine_tuned_vgg16_second_model.h5", custom_objects = {'f1_m' : f1_m})
    print('Model Loaded')
    return model 

def get_img_array(image, size = (224, 224)):
    # resize image
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    image_array = np.asarray(image)
    if len(image_array.shape) == 3:
        normalized_image_array = (image_array.astype(np.float32)/255.) # normalize the numpy array
        normalized_image_array = normalized_image_array[np.newaxis, ...]
    else:
        image_array =  np.stack((image_array, )*3, axis = -1)
        normalized_image_array = (image_array.astype(np.float32)/255.) # normalize the numpy array
        normalized_image_array = normalized_image_array[np.newaxis, ...]
    return normalized_image_array

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
def make_gradcam_heatmap(img_array):
    model = get_model()
    last_conv_layer_name = 'block5_conv3'
    classifier_layer_names = [
                              'block5_pool',
                              'global_average_pooling2d',
                              'flatten',
                              'dense',
                              'dropout',
                              'dense_1',
                              'dropout_1',
                              'dense_2',
                              'dropout_2',
                              'visualized_layer',
    ]
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def display_grad_img(image, heatmap):
    img = image
    img = img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = array_to_img(superimposed_img)

    # Display Grad CAM
    return superimposed_img