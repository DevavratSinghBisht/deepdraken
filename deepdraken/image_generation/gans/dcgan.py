from typing import Union
import numpy as np
from statistics import mean
import itertools

import torch
from torch import nn
import torchvision as tv

from deepdraken.base import BaseModel

# TODO class for dataloader

class DCGAN(BaseModel):

    def __init__(self,
                 net_G,
                 net_D,
                 device='cpu',
                 gpu_ids=None):
        
        super().__init__(device, gpu_ids)

        self.net_G = net_G
        self.net_D = net_D

        self.net_G.to(self.device)
        self.net_D.to(self.device)
        
        if 'cuda' in str(device):
            self.net_G = nn.DataParallel(self.net_G, self.gpu_ids)
            self.net_D = nn.DataParallel(self.net_D, self.gpu_ids)
        
        self.z_dim = next(self.net_G.parameters()).shape[-1]
        
        with torch.no_grad():
            self.c, self.w, self.h = self.net_G(self.get_noise(1, self.z_dim)).shape[1:]
        
        print(f'Detected Data:',
                f'\nNoise dimention    : {self.z_dim}', 
                f'\nNumber of Channels : {self.c}',
                f'\nImage Height       : {self.h}',
                f'\nImage Width        : {self.w}')

        self.model_names = ['net_G', 'net_D']
        self.optimizer_names = ['optimizer_G', 'optimizer_D']
        self.loss_names = ['loss_G', 'loss_D', 'loss_D_real', 'loss_D_fake']
        self.metric_names = []
        self.optimizers = []
        
        self.loss_G = float('nan')
        self.loss_D, self.loss_D_real, self.loss_D_fake = float('nan'), float('nan'), float('nan')

        self.history = {}
        self.batch_history = {}
        for loss_name in self.loss_names:
            self.history[loss_name] = []
            self.batch_history[loss_name] = []

    def set_data(self, batch) -> None:

        '''
        Unpacks a single batch data from the dataloader and perform necessary steps.
        Takes in a single batch as input.
        :param batch: single batch of data
        :return: None
        '''
        
        real_images, labels = batch
        self.real_images = real_images.to(self.device)
        self.labels = labels.to(self.device)
        self.batch_size = real_images.shape[0]
        
        self.valid = torch.ones((self.batch_size, 1)).to(self.device)
        self.fake = torch.zeros_like(self.valid)
        self.noise = self.get_noise(self.batch_size, self.z_dim)

    def get_noise(self, batch_size: int, z_dim: int) -> torch.Tensor:
        return torch.randn(batch_size, z_dim).to(self.device)

    def get_dataloader(self, root, batch_size, shuffle = True, transform = None):

        def get_default_transform():
            transform = []

            if self.c == 1:
                transform.append(tv.transforms.Grayscale(num_output_channels=1))

            transform.extend([tv.transforms.Resize((self.h, self.w)),
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.5,) * self.c, (0.5,) * self.c),])
            return tv.transforms.Compose(transform)

        if transform == None :
            transform = get_default_transform()
        
        dataset = tv.datasets.ImageFolder(root=root, transform=transform)
        print(f'Class to index mapping:\n{dataset.class_to_idx}')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def forward(self) -> None:
        '''
        Executes forward pass; called by both functions <optimize_parameters> and <test>.
        :return: None
        '''
        
        self.fake_images = self.net_G(self.noise)

    def backward_G(self) -> None:
        '''
        Executes backward pass for generator.
        '''
        
        self.loss_G = self.criterion(self.net_D(self.fake_images), self.valid)
        self.loss_G.backward()

    def backward_D(self) -> None:
        '''
        Executes backward pass for discriminator.
        '''
        
        self.loss_D_real = self.criterion(self.net_D(self.real_images), self.valid)
        self.loss_D_fake = self.criterion(self.net_D(self.fake_images.detach()), self.fake)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2
        self.loss_D.backward()

    def optimize_parameters(self) -> None:
        '''
        Calculate losses, gradients, and update network weights; called in every training iteration
        :return: None
        '''
        
        self.forward()      # compute fake images

        # -----------------
        #  Train Generator
        # -----------------
        self.set_requires_grad([self.net_D], False)     # D require no gradients when optimizing G
        self.optimizer_G.zero_grad()    # set G's gradients to zero
        self.backward_G()               # calculate gradients for G
        self.optimizer_G.step()         # update G's weights

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.set_requires_grad([self.net_D], True)
        self.optimizer_D.zero_grad()    # set D's gradients to zero
        self.backward_D()               # calculate gradients for D
        self.optimizer_D.step()         # update D's weights

    def compile(self,
                optim_G, 
                optim_D,
                criterion = nn.BCELoss()) -> None:
        '''
        Method for setting the optimizers, lr schedulers, etc.
        
        :param gen_optim: generator optimizer class
        :param disc_optim: discriminator optimizer class
        :param criterion: loss criterion, default: Binary Crossentropy
        :return: None
        '''

        self.optimizer_G = optim_G
        self.optimizer_D = optim_D
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.criterion = criterion.to(self.device)

    def fit(self, 
            data : Union[str, torch.Tensor], 
            epochs: int, 
            batch_size: int = 64, 
            shuffle: bool = True, 
            transform: tv.transforms.Compose = None,) -> None:
        '''
        Trains the models
        :param data: dataloader | path to dataset | pytorch tensor
        :param epochs (int): number of epochs to train the model
        :param batch_size (int): batch size to use while training, default: 64
        :param shuffle (bool): to shuffle the dataset or not, default: True
        :param transform : transformations to be applied on the dataset, i.e. preprocessing, default: None

        Note: batch_size , shuffle and transforms are applicable only when path to datset or tensor is proved as data
              in case dataloader is provided these parameters are handeled by dataloader itself.

        :return: None
        '''

        # Setting nets to training mode
        self.set_mode('train')

        if isinstance(data, str):
            print(f'String input detected.\nCreating a Dataloader using {data} directory')
            data = self.get_dataloader(data, batch_size, shuffle, transform)
        elif isinstance(data, torch.Tensor):
            # TODO for direct tensor inputs
            raise NotImplementedError()

        for epoch in range(epochs): # TODO use TQDM here
            self.on_epoch_start()
            
            for batch_idx, batch in enumerate(data):
                self.set_data(batch)
                self.optimize_parameters()
                self.append_batch_history() 

            self.append_history()
            print("[Epoch {}/{}] [G loss: {}] [D loss: {}]".format(epoch, epochs, self.history['loss_G'][-1], self.history['loss_D'][-1]))
            self.on_epoch_end()

    def generate_image(self, num_sample:int = 1, dtype:str ='tensor') -> Union[torch.Tensor, np.ndarray]:
        '''
        Generates images using Generator model.
        :param num_sample (int): number of images to be generated
        :param type (str): datatype of the returned image
                                -- tensor : pytorch tensor
                                -- numpy  : numpy array
        :return (torch.Tensor or numpy.ndarray): generated image pytorh tensor or numpy array
        '''

        noise = self.get_noise(num_sample, self.z_dim)
        with torch.no_grad():
            img = self.net_G(noise)

        img = img.cpu()

        if dtype == "tensor":
            return img
        elif dtype == "array":
            return img.numpy()
        else:
            print(f"Image of {dtype} datatype can not be generaed. Please choose from tensor or array")

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def append_batch_history(self):
        for name in itertools.chain(self.loss_names, self.metric_names):
            self.batch_history[name].append(float(getattr(self, name)))

    def append_history(self):

        for name in itertools.chain(self.loss_names, self.metric_names):
            self.history[name].append(mean(self.batch_history[name]))
            self.batch_history[name] = []

# TODO functionality of using lr schedulers
# TODO functionality of using data other than from dataloader, e.g. tensors
# TODO interpolating generated images --> linear, spherical
# TODO saving images in disc
# TODO loading only generator model for sampling if training mode not available.
# TODO functionality for custom noise function
# TODO gradient clipping
# TODO gradient penalty
