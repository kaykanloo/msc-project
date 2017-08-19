import sys 
import os
sys.path.append(os.path.join(os.getcwd(), '../Code/'))
from LadickyDataset import *

import tensorflow as tf
from keras.models import  Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input , Flatten, Dense, Reshape, Lambda
from keras.layers.convolutional import Conv2D

from math import ceil
from PIL import Image

def show_image(npimg):
    return Image.fromarray(npimg.astype(np.uint8))
def show_normals(npnorms):
    return Image.fromarray(((npnorms+1)/2*255).astype(np.uint8))

def mean_dot_product(y_true, y_pred):
    dot = tf.einsum('ijkl,ijkl->ijk', y_true, y_pred) # Dot product
    n = tf.cast(tf.count_nonzero(dot),tf.float32)
    mean = tf.reduce_sum(dot) / n
    return -1 * mean

def vgg16_model():
    # create model
    input_tensor = Input(shape=(240, 320, 3)) 
    base_model = VGG16(input_tensor=input_tensor,weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(80*60*3, activation='relu', name='fc2')(x)
    x = Reshape((60,80,3))(x)
    x = Lambda(lambda x: tf.image.resize_bilinear(x , [240,320]) )(x)
    pred = Lambda(lambda x: tf.nn.l2_normalize(x, 3) )(x)
    model = Model(inputs=base_model.input, outputs=pred)
    # Compile model
    model.compile(loss= mean_dot_product, optimizer='sgd')
    return model

file = '../Data/LadickyDataset.mat'
dataset = LadickyDataset(file)
model = vgg16_model()

batchSize = 32
epochs = 1
totalBatches = epochs * ceil(dataset.size/batchSize)

for batch in range(totalBatches):
    imgs, norms = dataset.get_batch(batchSize)
    model.fit(imgs, norms, batch_size=batchSize, epochs=1)