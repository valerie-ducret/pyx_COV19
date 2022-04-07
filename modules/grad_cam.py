# needed packages

#from tensorflow import keras

from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.cm as cm
import numpy as np
import pandas as pd 

from IPython.display import Image, display

import tensorflow as tf
import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras import layers
from keras.models import Model, Sequential
from keras.metrics import Recall, Precision, AUC
from keras.applications.vgg16 import VGG16
from keras.models import load_model

preprocess_input = keras.applications.xception.preprocess_input

def gradcam(img_path, model):
    size = (224,224)
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis = 0)
    array.shape
    img_array = preprocess_input(array)
    
    # This is our convolutional and classifier layers. To see yours, just print model.summary(), look at the last "_conv", 
    # and the following ones are the classifier layers
    # to have a function to retrieve the classifier or convolutional layers (adaptable with "if 'conv' not in layer.name")
    #classifier_layer_names = []
    #for i in range(18,28):
    #    layer = model.layers[i]
    #    # check for non-convolutional layer
    #    if 'conv' in layer.name:
    #        continue
    #    classifier_layer_names.append(layer.name)
    last_conv_layer_name = "block5_conv3"
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
    
    size_2 = (512,512)
    img = load_img(img_path, target_size = size_2)
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
    display(superimposed_img)

