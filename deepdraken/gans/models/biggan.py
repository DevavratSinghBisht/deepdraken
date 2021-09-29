import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from ...utils import one_hot, one_hot_if_needed
from utils import truncated_noise_sample, interpolate_and_shape

class BigGAN():

    '''
        Class for generating big gan images.
        Note: The class is still in development it may further change a lot.
    '''
    MODLES = { 'biggan-deep-128': 'https://tfhub.dev/deepmind/biggan-deep-128/1',  # 128x128 BigGAN-deep
               'biggan-deep-256': 'https://tfhub.dev/deepmind/biggan-deep-256/1',  # 256x256 BigGAN-deep
               'biggan-deep-512': 'https://tfhub.dev/deepmind/biggan-deep-512/1',  # 512x512 BigGAN-deep
               'biggan-128': 'https://tfhub.dev/deepmind/biggan-128/2',  # 128x128 BigGAN
               'biggan-256': 'https://tfhub.dev/deepmind/biggan-256/2',  # 256x256 BigGAN
               'biggan-512': 'https://tfhub.dev/deepmind/biggan-512/2'}  # 512x512 BigGAN

    def __init__(self, model_name = 'biggan-deep-512') -> None:

        self.model = hub.KerasLayer(self.MODLES[model_name])

    def __get_images(self, y, z, truncation):
        # generating and post processing the image
        images = self.model({'y': y, 'z':z, 'truncation': truncation})
        images = np.asarray(images)
        images = np.clip(((images + 1) / 2.0) * 256, 0, 255)
        images = np.uint8(images)
        return images

    def sample(self, num_samples, label, truncation=0.4, noise_seed=None):
        '''
        Generates Images

        :param num_samples: number of images to generate
        :param label: class of the image to be generated
        :param truncation: # TODO fill here
        :param noise_seed: seed for generating the noise, random seed is taken if None
        :return: an array of generated images
        '''

        # generating the noise
        z = truncated_noise_sample(num_samples, 128, truncation, noise_seed)
        # converting into array if needed
        z = np.asarray(z)
        label = np.asarray(label)
        
        # if an integer is provided as label
        # create an array of labels
        if len(label.shape) == 0:
            label = np.asarray([label] * num_samples)
        
        # if number of labels is not equal to number of samples to be generated
        if label.shape[0] != num_samples:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'.format(z.shape[0], label.shape[0]))
        
        y = one_hot_if_needed(label, 1000) # one hot encoding the label

        
        images = self.__get_images(y, z, truncation)
        return images

    def interpolate(self, num_samples, label_A, label_B, num_interps=5, truncation=0.2, noise_seed_A=None, noise_seed_B=None):
        '''
        Generates interpolations for images between two classes or labels.

        :param num_samples: number of samples to generate
        :param label_A: class 1 for interpolation
        :param  label_A: class 2 for interpolation
        :param num_interps: number of interpolations to generate between the classes
        :param truncation: # TODO fill here
        :param noise_seed_A: seed for generating noise for the 1st class
        :param noise_seed_B: seed for generating noise for the 2nd class
        :return: an array of array containing interpolated images
        '''

        z_A, z_B = [truncated_noise_sample(num_samples, 128, truncation, noise_seed) for noise_seed in [noise_seed_A, noise_seed_B]]
        y_A, y_B = [one_hot([category] * num_samples, 1000) for category in [label_A, label_B]]

        z_interp = interpolate_and_shape(z_A, z_B, num_interps)
        y_interp = interpolate_and_shape(y_A, y_B, num_interps)

        images = self.__get_images(y_interp, z_interp, truncation)
        shape = [num_samples, num_interps]
        shape.extend([i for i in images.shape[1:]])
        images = images.reshape(shape)
        return images
