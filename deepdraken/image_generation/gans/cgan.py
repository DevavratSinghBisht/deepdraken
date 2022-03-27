from typing import List
from pathlib import Path

import torch
import torchvision as tv

from deepdraken.image_generation.gans.dcgan import DCGAN

class CGAN(DCGAN):

    def __init__(self,
                 gen,
                 disc,
                 n_classes:int,
                 device : str = 'cpu',
                 gpu_ids : List[int] = None):
                 
        super().__init__(gen, disc, device, gpu_ids)
        self.n_classes = n_classes

    def set_data(self, batch):
        super().set_data(batch)
        self.fake_labels = torch.randint(0, self.n_classes, (self.batch_size,) , device=self.device)

    def forward(self):
        
        self.fake_images = self.net_G(self.noise, self.labels)

    def backward_G(self):
        
        self.loss_G = self.criterion(self.net_D(self.fake_images, self.fake_labels), self.valid)
        self.loss_G.backward()

    def backward_D(self):
        
        self.loss_D_real = self.criterion(self.net_D(self.real_images, self.labels), self.valid)
        self.loss_D_fake = self.criterion(self.net_D(self.fake_images.detach(), self.fake_labels.detach()), self.fake)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2
        self.loss_D.backward()