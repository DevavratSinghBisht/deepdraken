import numpy as np
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
    :param truncation: # TODO fill here
    :param seed: random seed to use
    :return: noise sample of given dimension and truncation
    '''

    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(n_samples, dim), random_state=state)
    return truncation * values
    