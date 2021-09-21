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

def get_discriminator_model(input_shape) -> tf.keras.Sequential :
    '''
        Creates and returns a Generator Model.
    '''
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
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
    generator = get_generator_model(noise_dim, n_channels)
    discriminator_input_dim = generator.output_shape[1:]
    discriminator = get_discriminator_model(discriminator_input_dim)
    return generator, discriminator


class GAN():

    '''
        Base class for Training DCGANs.

    '''
    
    def __init__(self, 
                 generator = None,
                 discriminator = None,
                 optimizer = None,
                 loss = 'binary crossentropy',
                 ) -> None:

        '''
        :param generator: generator model or path to the generator model
        :param discriminator: discriminator model path to the discriminator model
        :param loss: loss function or the name of loss function
        :param optimizer: optimizer or the name of the optimizer
        
        :returns: None
        '''

        if type(loss) == str: 
            self.loss = self.get_loss(loss)
        else:
            self.loss = loss

        if type(generator) == str and type(discriminator) == str:
            self.generator, self.discriminator = self.load_model(generator, discriminator)
        elif generator == None and discriminator == None:
            self.image_dim = (28 ,28, 3)
            self.noise_dim = 128
            self.generator, self.discriminator = get_generator_discriminator_pair(self.noise_dim, self.image_dim[-1])
        else:
            self.assert_models(generator, discriminator)

        self.generator_optimizer, self.discriminator_optimizer = self.get_optimizer(optimizer)

    def load_model(self, generator_path: str, discriminator_path: str):

        '''
            Loads the generator and discriminator models.

            :param generator_path: path to the generator model
            :param discriminator_path: path to the discriminator model
            
            :return: None
        '''
        generator = tf.keras.models.load_model(generator_path)
        discriminator = tf.keras.models.load_model(discriminator_path)

        self.assert_models(generator, discriminator)

        return generator, discriminator

    def assert_models(self, generator, discriminator) -> None:

        '''
            Checks if the models are compatible.

            :param generator: generator model
            :param discriminator: discriminator model

            :return: None
        '''
        #TODO generalise the noise dimension
        noise_dim = generator.input_shape
        assert len(noise_dim) == 2, "Noise Vector should have a single dimension."
        self.noise_dim = noise_dim[1]

        #TODO generalise the image dimension
        generator_output_dim = generator.output_shape
        assert len(generator_output_dim) == 4 , "Output Image is not 3 dimensional"
        self.image_dim = generator_output_dim[1:]

        discriminator_input_dim = discriminator.input_shape
        assert generator_output_dim == discriminator_input_dim, "Generator output doens't have the same shape as discriminator input."

    def get_loss(self, loss: str):
        '''
            Sets the loss function that will be used for calculated Generator and Discriminator Loss 
        '''
        if loss == 'binary crossentropy' or loss == 'bce':
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif loss == 'wasserstein':
            # TODO return wasserstein loss
            raise NotImplementedError(('wasserstein loss not implemented'))

    def generator_loss(self, fake_output):
        '''
            Calculates Generator Loss.
        '''
        return self.loss(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        '''
            Calculates Discriminator Loss.
        '''

        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def get_optimizer(self, optimizer=None) -> tuple:
        '''
            :param optimizer: name of optimzer or a an object from any of the classes in keras.optimizers or a list of these attributes
            :return: tuple of generator optimizer and discriminator optimizer
        '''


        def get_optim(optimizer, dtype_optim):
            
            if dtype_optim == str:
                tf.keras.optimizers.get(optimizer)
            else:
                return optimizer
            
        if type(optimizer) == list:
            # TODO better assert string
            assert len(optimizer) == 2, 'The list must contain two elements, one for generator and one for discriminator'
            return get_optim(optimizer[0]), get_optim(optimizer[1]) 

        elif optimizer == None:
            return tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4)
        else:
            optim = get_optim(optimizer)
            return optim, optim

    # `tf.function` causes the function to be compiled.
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

    def train(self, data_path, batch_size, epochs, shuffle = True) -> None:
        '''
            Trains the model.

            :param data_path: path to the dataset for training
            :param batch_size: batch size for training
            :param epochs: number of epochs to train on
            :param shuffle: to shuffle the dataset or not while training

            :return: None
        '''

        datagen = GANDataGenerator(data_path, batch_size, self.image_dim, shuffle)
        
        for epochs in tqdm(range(epochs)):
            for image_batch in datagen:
                self.__train_step(image_batch, batch_size)

    def run(self, data_path, epochs=2, batch_size=2, shuffle=True) -> None:
        
        '''
            Calls all other functions in order to train the models.
        '''
    
        self.train(data_path, batch_size, shuffle, epochs)
