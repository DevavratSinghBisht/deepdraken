import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deepdraken.experimental.gans.pretrained.big_gan import BigGAN
from deepdraken.experimental.gans.pretrained.compare_gan import CompareGAN