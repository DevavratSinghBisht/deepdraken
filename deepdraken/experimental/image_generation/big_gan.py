from typing import Optional, Union, List
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from deepdraken.utils import one_hot, one_hot_if_needed, truncated_noise_sample, interpolate_and_shape


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
               label: List[int],
               truncation: Optional[float] = 0.4,
               noise_seed: Optional[int] = None):
        '''
        Method for sampling images

        :param label: class of the image to be generated
        :param num_samples: number of images to generate
        :param truncation: truncation value for the noise vector
        :param noise_seed: seed for generating the noise, random seed is taken if None
        :param plot: plots the images if set to true
        :return: an array of generated images
        '''
        num_samples = len(label)
    
        # generating the noise
        z = truncated_noise_sample(num_samples, 128, truncation, noise_seed)
        # converting into array if needed
        z = np.asarray(z)
        label = np.asarray(label)
        
        y = one_hot_if_needed(label, 1000) # one hot encoding the label
        
        images = self.__get_images(y, z, truncation)

        return images

    def interpolate(self,
                    label_A: str,
                    label_B: str,
                    num_samples: int,
                    num_interps: Optional[int] = 5,
                    truncation: Optional[float] = 0.2,
                    noise_seed_A: Optional[int] = None,
                    noise_seed_B: Optional[int] = None):
        '''
        Generates interpolations for images between two classes or labels.
        
        :param label_A: class 1 for interpolation
        :param label_B: class 2 for interpolation
        :param num_samples: number of samples to generate
        :param num_interps: number of interpolations to generate between the classes
        :param truncation: truncation value for the noise vector
        :param noise_seed_A: seed for generating noise for the 1st class
        :param noise_seed_B: seed for generating noise for the 2nd class
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

        return images