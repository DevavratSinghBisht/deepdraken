from typing import Optional, Union

import tensorflow as tf
import tensorflow_hub as hub

from ...utils import one_hot, one_hot_if_needed
from ..utils import truncated_noise_sample, interpolate_and_shape

import numpy as np
import matplotlib.pyplot as plt
import math

class BigGAN():

    '''
        Class for generating big gan images.
    '''
    MODELS = { 'biggan-deep-128': 'https://tfhub.dev/deepmind/biggan-deep-128/1',  # 128x128 BigGAN-deep
               'biggan-deep-256': 'https://tfhub.dev/deepmind/biggan-deep-256/1',  # 256x256 BigGAN-deep
               'biggan-deep-512': 'https://tfhub.dev/deepmind/biggan-deep-512/1',  # 512x512 BigGAN-deep
               'biggan-128': 'https://tfhub.dev/deepmind/biggan-128/2',  # 128x128 BigGAN
               'biggan-256': 'https://tfhub.dev/deepmind/biggan-256/2',  # 256x256 BigGAN
               'biggan-512': 'https://tfhub.dev/deepmind/biggan-512/2'}  # 512x512 BigGAN

    def __init__(self, model_name: Optional[str] = 'biggan-deep-512') -> None:

        self.model = hub.KerasLayer(self.MODELS[model_name])

    def __get_images(self, y: np.ndarray, z: np.ndarray, truncation: int) -> np.ndarray:
        '''
        Method for generating images.
        :param y: one hot encoded class labels
        :param z: noise vectors
        :param truncations: truncation value for the noise vector
        :return: generated images
        '''
        # generating and post processing the image
        images = self.model({'y': y, 'z':z, 'truncation': truncation})
        images = np.asarray(images)
        images = np.clip(((images + 1) / 2.0) * 256, 0, 255)
        images = np.uint8(images)
        return images

    def sample(self,
               label: Union[int, list, tuple],
               num_samples: Optional[int] = 1,
               truncation: Optional[float] = 0.4,
               noise_seed: Optional[int] = None,
               plot: Optional[bool] = False):
        '''
        Method for sampling images

        :param label: class of the image to be generated
        :param num_samples: number of images to generate
        :param truncation: truncation value for the noise vector
        :param noise_seed: seed for generating the noise, random seed is taken if None
        :param plot: plots the images if set to true
        :return: an array of generated images
        '''

        if type(label) == list or type(label) == tuple:
            num_samples = len(label)

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

        if plot:
            self.plot(images)

        return images

    def interpolate(self,
                    label_A: int,
                    label_B: int,
                    num_samples: int,
                    num_interps: Optional[int] = 5,
                    truncation: Optional[float] = 0.2,
                    noise_seed_A: Optional[int] = None,
                    noise_seed_B: Optional[int] = None,
                    plot: Optional[bool] = False):
        '''
        Generates interpolations for images between two classes or labels.
        
        :param label_A: class 1 for interpolation
        :param label_B: class 2 for interpolation
        :param num_samples: number of samples to generate
        :param num_interps: number of interpolations to generate between the classes
        :param truncation: truncation value for the noise vector
        :param noise_seed_A: seed for generating noise for the 1st class
        :param noise_seed_B: seed for generating noise for the 2nd class
        :param plot: if true then plots the generated images
        :return: an array of array containing interpolated images
        '''

        # generating noise samples of shape num_samples, 128 each
        z_A, z_B = [truncated_noise_sample(num_samples, 128, truncation, noise_seed) for noise_seed in [noise_seed_A, noise_seed_B]]
        # generating one_hot encoded class vectors of the class
        y_A, y_B = [one_hot([category] * num_samples, 1000) for category in [label_A, label_B]]

        # interpolating the noise samples and class vectors
        z_interp = interpolate_and_shape(z_A, z_B, num_samples, num_interps)
        y_interp = interpolate_and_shape(y_A, y_B, num_samples, num_interps)

        # generating and reshaping the image
        images = self.__get_images(y_interp, z_interp, truncation)
        shape = [num_samples, num_interps]
        shape.extend([i for i in images.shape[1:]])
        images = images.reshape(shape)

        if plot:
            self.plot(images)

        return images

    def plot(self, images: np.ndarray, columns: Optional[int] = None) -> None:
        '''
        Method for plotting images.
        The input array can have 5 dimensions as (row, column, h, w, c)
        or have 4 dimensions as (number of images, h, w, c)
        If number of columns is not provided then:
            1. taken as 2nd dimension for 5 dimensional array
            2. taken as 3 for 4 dimensional array

        :param images: numpy array containing the generated images
        :param columns: number of columns in the plot
        :return: None
        '''
        
        if columns != None:
            if len(images.shape) == 5: # images from interpolating
                num_images = images.shape[0] * images.shape[1]

            elif len(images.shape) == 4: # images from sampling
                num_images = images.shape[0]
            
            col = columns
            row = (num_images // col) + 1

        else:
            if len(images.shape) == 5: # images from interpolating
                row = images.shape[0] # number of samples
                col = images.shape[1] # number of interpolations

            elif len(images.shape) == 4: # images from sampling
                num_images = images.shape[0]
                col = 3 if num_images > 3 else num_images
                row = math.ceil(num_images / col)

        _, axs = plt.subplots(row, col, figsize=(12, 12))
        axs = axs.flatten()
        imgs = np.reshape(images, [row * col, *images.shape[-3:]]) 
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
        plt.show()
