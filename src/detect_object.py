#
# feature_extractor.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np


def get_model():
    """Load VGG16 model and remove last two layers for feature extraction"""
    vgg_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
    
    # Remove last two layers (softmax classification layers)
    # This keeps the 4096-dimensional feature vector
    layers = [layer for layer in vgg_model.layers[:-2]]
    
    inp = vgg_model.input
    out = vgg_model.layers[-3].output  # Get output from fc2 layer (4096 features)
    
    model = Model(inp, out)
    return model

def get_features(model, cropped_image):
    """Extract features from an image using the VGG16 model"""
    x = image.img_to_array(cropped_image)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features
