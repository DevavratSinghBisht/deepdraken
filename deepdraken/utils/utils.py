import numpy as np

def one_hot(class_idx: int, total_classes: int) -> np.ndarray:
    '''
        Performs one hot enoding on the index.
        
        :param class_idx: index of the selected class
        :param total_classes: total number of classes
        :returns: 2D aray of one hot encoded indices.
    '''
    
    index = np.asarray(class_idx)
    
    if len(index.shape) == 0:
        index = np.asarray([index])
    
    assert len(index.shape) == 1
    
    num = index.shape[0]
    output = np.zeros((num, total_classes), dtype=np.float32)
    output[np.arange(num), index] = 1
    
    return output

def one_hot_if_needed(class_idx, total_classes) -> np.ndarray:
    '''
        Checks if the labels are already one hot encoded.
        Performs One hot encoding only if the labels are not already onehot encoded.

        :param class_idx: index of the selected class
        :param total_classes: total number of classes
        :returns: 2D aray of one hot encoded indices.
    '''
    label = np.asarray(class_idx)
    
    if len(label.shape) <= 1:
        label = one_hot(class_idx, total_classes)
    
    assert len(label.shape) == 2
    return label

## GAN Utils

from scipy.stats import truncnorm


def noise_sample(n_samples, dim, limits = (-1, 1)):
    '''
    Generates noise vector of given dimension.

    :param n_samples: number of noise vector samples to be generated
    :param dim: dimension of the noise vector to be generated
    :param limits: lower and upper limits of the values in the noise vector
    :return: noise sample of given dimension
    '''
    noise = np.random.uniform(limits[0], limits[1], size=(n_samples, dim))
    return noise


def interpolate(A, B, num_interps):
    '''
        Find interpolations between two given vectors.

        :param A: 1st vector
        :param B: 2nd vector
        :param num_interps: number of interpolations to make
        :return: an array of interpolated noise vectors
    '''
    
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    alphas = np.linspace(0, 1, num_interps)
    return np.array([(1-a)*A + a*B for a in alphas])

def interpolate_and_shape(A, B, num_samples, num_interps):
    interps = interpolate(A, B, num_interps)
    return (interps.transpose(1, 0, *range(2, len(interps.shape))).reshape(num_samples * num_interps, *interps.shape[2:]))


def truncated_noise_sample(n_samples, dim, truncation=1., seed=None):
    '''
    Generates truncated noise samples.
    
    :param n_samples: number of noise vector samples to be generated
    :param dim: dimension of the noise vector to be generated
    :param truncation: truncation value for the noise vector
    :param seed: random seed to use
    :return: noise sample of given dimension and truncation
    '''

    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(n_samples, dim), random_state=state)
    return truncation * values
    
# Experimental GAN utils