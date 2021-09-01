''' 
    This file conatins the base class for DCGAN and related functions.
'''

from numpy.lib.function_base import kaiser
import tensorflow as tf
from tensorflow.keras import layers

from ..data_prep.data_gen import DataGenerator
from .models.scartch import get_generator_discriminator_pair

class DCGAN():

    '''
        Base class for DCGANs. This calss will be inherited by other classes.

    '''
    
    def __init__(self, 
                 noise_dim, 
                 data_path,
                 batch_size=32,
                 dim = (28, 28, 3), 
                 shuffle=True, 
                 **kwargs) -> None:


        self.data_path = data_path
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        self.data_loader(data_path, batch_size, dim, shuffle)
        self.get_generator_discriminator(kwargs.get('generator'), kwargs.get('discriminator'))

        # This method makes a helper function to compute cross entropy loss
        self.set_loss()
        self.set_optimizer()

    def data_loader(self, data_path, batch_size, dim, shuffle) -> None:
        '''
            Preprocess and Loads the dataset.
        '''
        self.datagen = DataGenerator(data_path, batch_size, dim, shuffle)

    def set_loss(self) -> None:
        '''
            Sets the loss function that will be used for calculated Generator and Discriminator Loss 
        '''
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        '''
            Calculates Discriminator Loss.
        '''

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output) -> None:
        '''
            Calculates Generator Loss.
        '''
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def set_optimizer(self) -> None:
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    # `tf.function` causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        '''
            Performs training step for one batch.
        '''

        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip (gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs) -> None:
        '''
            Trains the model.
        '''
        for epoch in range(epochs):

            for image_batch in dataset:
                self.train_step(image_batch)

    def get_generator_discriminator(self, generator, discriminator, noise_dim, n_channels):
        
        if generator == None and discriminator == None:
            print("Creating scratch Generator and Discriminator as none were provided.")
            self.generator, self.discriminator = get_generator_discriminator_pair(noise_dim, n_channels)
        
        elif generator == None and discriminator != None :
            pass

        elif generator != None and discriminator == None :
            pass

        else:
            self.generator = generator
            self.discriminator = discriminator

    def run(self, epochs, **kwargs) -> None:
        
        '''
            Calls all other functions in order to train the models.
        '''

        if (kwargs.get('batch_size') and kwargs.get('data_path') and kwargs.get('shuffle')) != None :
        
            if kwargs.get('batch_size') is not None:
                self.batch_size = kwargs.get('batch_size')
    
            if kwargs.get('data_path') is not None:
                self.batch_size = kwargs.get('data_path')

            if kwargs.get('shuffle') is not None:
                shuffle = kwargs.get('shuffle')
            else:
                shuffle = True

            self.data_loader(self.data_path, self.batch_size, self.dim, shuffle)

        self.train(self.datagen, epochs)
