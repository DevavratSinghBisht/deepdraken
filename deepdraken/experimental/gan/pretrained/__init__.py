import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .big_gan import BigGAN
from .compare_gan import CompareGAN