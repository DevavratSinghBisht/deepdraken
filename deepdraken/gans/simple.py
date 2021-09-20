''' 
    This file conatins a class and functions for GAN Training.
'''
from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers
from ..data_prep.data_gen import GANDataGenerator
from tqdm import tqdm


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


class GAN():

    '''
        Base class for Training DCGANs.

    '''
    
    def __init__(self, 
                 optimizer,
                 generator_path,
                 discriminator_path,
                 ) -> None:

        '''
        :param optimizer: optimizer or the name of the optimizer
        :param generator_path: path to the generator model
        :param discriminator_path: path to the discriminator model
        
        :returns: None
        '''

        # TODO if string inputs are given then get their respective objects
        self.set_loss()
        self.set_optimizer()

        #self.generator
        #self.discriminator
        #self.generator_optimizer
        #self.discriminator_optimizer
        #self.image_dim
        #self.noise_dim

    def load_model(self, generator_path: str, discriminator_path: str) -> None:

        '''
            Loads the generator and discriminator models.

            :param generator_path: path to the generator model
            :param discriminator_path: path to the discriminator model

            :return: None
        '''
        
        # TODO use the paths to load the model 
        self.generator = generator_path
        self.discriminator = discriminator_path

        # TODO use the model arcitecture to find the noise dim size and the image dim size

    def create_model(self, noise_dim: Tuple, image_dim: Tuple) -> None:

        self.noise_dim = noise_dim
        self.image_dim = image_dim

        # TODO create a generator discriminator pair using the noise dim and image dim        
        print("Creating scratch Generator and Discriminator pair.")
        self.generator, self.discriminator = get_generator_discriminator_pair(noise_dim, image_dim)

    def set_loss(self) -> None:
        '''
            Sets the loss function that will be used for calculated Generator and Discriminator Loss 
        '''
        # TODO set loss uing the the string provided in __init__
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        '''
            Calculates Discriminator Loss.
        '''

        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output) -> None:
        '''
            Calculates Generator Loss.
        '''
        return self.loss(tf.ones_like(fake_output), fake_output)

    def set_optimizer(self) -> None:

        # TODO set optimizer using the string provided in __init__
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    # `tf.function` causes the function to be "compiled".
    @tf.function
    def __train_step(self, images, batch_size) -> None:
        '''
            Performs training step for one batch.

            :param images:
            :param batch_size:

            :return: None
        '''

        # generating noise samples from a random normal distribution
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # generating fake images
            generated_images = self.generator(noise, training=True)

            # generating predictions for real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # calculating generator and discriminator loss
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # getting gradients for generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # applying the gradients to respective models
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip (gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, data_path, batch_size, dim, shuffle, epochs) -> None:
        '''
            Trains the model.

            :param data_path: path to the dataset for training
            :param batch_size: batch size for training
            :param dim: dimension of the image on which the GAN will be trained
            :param shuffle: to shuffle the dataset or not while training
            :param epochs: number of epochs to train on

            :return: None
        '''

        datagen = GANDataGenerator(data_path, batch_size, dim, shuffle)
        
        for epochs in tqdm(range(epochs)):
            for image_batch in datagen:
                self.__train_step(image_batch, batch_size)

    def run(self, epochs, data_path, batch_size, shuffle=True) -> None:
        
        '''
            Calls all other functions in order to train the models.
        '''
        self.create_model(128, 3)

        self.train(data_path, batch_size, (28, 28, 3), shuffle, epochs)
