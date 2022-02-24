from typing import Optional, Union

import tensorflow as tf
import tensorflow_hub as hub

from ....utils import noise_sample

import numpy as np

class CompareGAN():

    '''
        Class for generating Compare GAN images.
    '''
    MODLES = {  'celeba-hq-1': 'https://tfhub.dev/google/compare_gan/model_1_celebahq128_resnet19/1',
                'celeba-hq-2': 'https://tfhub.dev/google/compare_gan/model_2_celebahq128_resnet19/1',
                'celeba-hq-3': 'https://tfhub.dev/google/compare_gan/model_5_celebahq128_resnet19/1',
                'celeba-hq-4': 'https://tfhub.dev/google/compare_gan/model_6_celebahq128_resnet19/1',
                'celeba-hq-5': 'https://tfhub.dev/google/compare_gan/model_9_celebahq128_resnet19/1',

                'lsun-bedroom-1':	'https://tfhub.dev/google/compare_gan/model_3_lsun_bedroom_resnet19/1',
                'lsun-bedroom-2':	'https://tfhub.dev/google/compare_gan/model_4_lsun_bedroom_resnet19/1',
                'lsun-bedroom-3': 'https://tfhub.dev/google/compare_gan/model_7_lsun_bedroom_resnet19/1',
                'lsun-bedroom-4': 'https://tfhub.dev/google/compare_gan/model_8_lsun_bedroom_resnet19/1',
                'lsun-bedroom-5': 'https://tfhub.dev/google/compare_gan/model_10_lsun_bedroom_resnet19/1',

                'cifar-10-1': 'https://tfhub.dev/google/compare_gan/model_11_cifar10_resnet_cifar/1',
                'cifar-10-2': 'https://tfhub.dev/google/compare_gan/model_12_cifar10_resnet_cifar/1',
                'cifar-10-3': 'https://tfhub.dev/google/compare_gan/model_13_cifar10_resnet_cifar/1',
                'cifar-10-4': 'https://tfhub.dev/google/compare_gan/model_14_cifar10_resnet_cifar/1',
                'cifar-10-5': 'https://tfhub.dev/google/compare_gan/model_15_cifar10_resnet_cifar/1'}

    def __init__(self, model_name: Optional[str] = 'cifar-10-1') -> None:

        self.model = hub.KerasLayer(self.MODLES[model_name], signature="generator")

    def __get_images(self, z: np.ndarray) -> np.ndarray:
        '''
        Method for generating images.
        :param z: noise vectors
        :return: generated images
        '''

        images = self.model({'z':z})
        images = np.asarray(images)
        images = np.clip(images * 255, 0, 255)
        images = np.uint8(images)
        return images

    def sample(self, num_samples:int) -> np.ndarray:
        '''
        Method for sampling images

        :param num_samples: number of images to generate
        :return: an array of generated images
        '''

        generated_samples = 0
        images = []


        while generated_samples < num_samples:
            # generating the noise
            z = noise_sample(64, 128) # compare gan expects the batch size to be 64
            # converting into array if needed
            z = np.asarray(z)

            image_batch = self.__get_images(z)
            images.append(image_batch)
            generated_samples += 64

        images = np.concatenate(images, axis=0)
        return images[:num_samples]