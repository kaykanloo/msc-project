
import warnings

import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

def model(input_shape=(240, 320, 3)):
    """Instantiates the baseline architecture.
    
    # Arguments
        input_shape: optional shape tuple,
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid input shape.
    """

    img_input = Input(shape=input_shape)

    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = Conv2D(3, (1, 1), activation='relu', padding='same', name='conv5')(x)

    # Top Layers
    x = Flatten()(x)
    x = Dense(80*60*3, activation='relu', name='fc1')(x)
    x = Reshape((60,80,3))(x)
    x = Lambda(lambda x: tf.image.resize_bilinear(x , [240,320]) )(x)
    x = Lambda(lambda x: tf.nn.l2_normalize(x, 3) )(x)
    
    # Create model.
    inputs = img_input
    model = Model(inputs, x, name='baseline')
    
    return model
