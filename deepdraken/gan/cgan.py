import numpy as np
import torch

from .dcgan import DCGAN

class CGAN(DCGAN):

    def __init__(self, gen, disc, device, is_train=True, gpu_ids=[]):
        super().__init__(gen, disc, device, is_train, gpu_ids)

    def set_data(self, batch):
        super().set_data(batch)
        self.fake_labels = torch.from_numpy(np.random.randint(0, self.n_classes, self.batch_size)).to(self.device) # TODO manage n_classes

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