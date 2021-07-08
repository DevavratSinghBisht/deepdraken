''' 
    This files helps in developing a GAN model from scratch.
'''

import tensorflow as tf
from tensorflow.keras import layers


def get_generator_model(noise_dim, n_channels) -> tf.keras.Sequential:
    '''
        Creates and returns a Generator Model.
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(n_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

def get_discriminator_model(n_channels) -> tf.keras.Sequential :
    '''
        Creates and returns a Generator Model.
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, n_channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def get_generator_discriminator_pair(noise_dim, n_channels) -> tuple:
    '''
        Creates a generator discriminator pair that are compatible with each other.
    '''
    return get_generator_model(noise_dim, n_channels), get_discriminator_model(n_channels)
