import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_path, batch_size=32, dim=(28,28,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.dataset_path = dataset_path
        self.list_IDs = os.listdir(dataset_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.load_img(self.dataset_path + "/" + ID)

        return X

    def load_img(self, path):
        h, w , c = self.dim
        img = cv2.imread(path)
        img = cv2.resize(img, (h, w))
        img = (img - 127.5) / 127.5
        return img