import tensorflow.compat.v1 as tf
import numpy as np
from scipy.stats import truncnorm
import tensorflow_hub as hub

class BigGAN():

    '''
        Class for generating big gan images.
        Note: The class is still in development it may further change a lot.
    '''

    def __init__(self, model_name) -> None:
        
        print("Disabling Tensorflow V2 Behaviour.")
        tf.disable_v2_behavior()

        self.BigGAN_deep_models = { 'biggan-deep-128': 'https://tfhub.dev/deepmind/biggan-deep-128/1',  # 128x128 BigGAN-deep
                                    'biggan-deep-256': 'https://tfhub.dev/deepmind/biggan-deep-256/1',  # 256x256 BigGAN-deep
                                    'biggan-deep-512': 'https://tfhub.dev/deepmind/biggan-deep-512/1',  # 512x512 BigGAN-deep
                                    'biggan-128': 'https://tfhub.dev/deepmind/biggan-128/2',  # 128x128 BigGAN
                                    'biggan-256': 'https://tfhub.dev/deepmind/biggan-256/2',  # 256x256 BigGAN
                                    'biggan-512': 'https://tfhub.dev/deepmind/biggan-512/2'}  # 512x512 BigGAN

        self.module = hub.Module(self.BigGAN_deep_models[model_name])
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                       for k, v in self.module.get_input_info_dict().items()}

        self.output = self.module(self.inputs)

        self.input_z = self.inputs['z']
        self.input_y = self.inputs['y']
        self.input_trunc = self.inputs['truncation']

        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]

        self.initializer = tf.global_variables_initializer()

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        '''
            Generates truncated noise samples.

            Params:
                batch_size:
                truncation: the value at which truncation is to be performed.
                seed: random seed, random value is selected if not provided.

            Returns:
                noise vectors
            
        '''

        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def sampler(self, noise, label, truncation=1., batch_size=8):
        '''
        
        '''

        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'.format(noise.shape[0], label.shape[0]))
        
        label = self.one_hot_if_needed(label)
        images = []

        with tf.Session() as sess:
            sess.run(self.initializer)
            for batch_start in range(0, num, batch_size):
                s = slice(batch_start, min(num, batch_start + batch_size))
                feed_dict = {self.input_z: noise[s], self.input_y: label[s], self.input_trunc: truncation}
                images.append(sess.run(self.output, feed_dict=feed_dict))
        
        images = np.concatenate(images, axis=0)
        assert images.shape[0] == num
        images = np.clip(((images + 1) / 2.0) * 256, 0, 255)
        images = np.uint8(images)
        
        return images

    def sample(self, class_label, num_samples, truncation=0.4, noise_seed=0):
        '''
            Runs the sampler function.
        '''
        z = self.truncated_z_sample(num_samples, truncation, noise_seed)
        y = class_label
        return self.sampler(z, y,truncation)