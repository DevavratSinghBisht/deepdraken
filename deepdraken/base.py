from collections import OrderedDict
from abc import ABC, abstractclassmethod, abstractmethod
from typing import List
from pathlib import Path

import torch
from torch import nn
from .utils.base_utils import get_scheduler

class BaseModel(ABC):
    """
    This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>               :   initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_data>               :   unpack data from dataset and apply preprocessing.
        -- <forward>                :   produce intermediate results.
        -- <optimize_parameters>    :   calculate losses, gradients, and update network weights.
        -- <complie>                :   intializes the optimizers
        -- <fit>                    :   trains the models
    """

    def __init__(self,
                 device : str = 'cpu',
                 gpu_ids: List[int] = None):
        """
        Initialize the BaseModel class.
        
        :param device (str): device (cpu or cuda) to use
        :param gpu_ids (list of int): CUDA devices (default: all devices) 
        
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, device, gpu_ids)>
        Then, you need to define four lists:
            -- self.model_names (str list)      :   specify the networks used in training.
            -- self.loss_names (str list)       :   specify the training losses that you want to plot and save.
            -- self.optimizer_names (str list)  :   specifgy the optimizers. 
            -- self.optimizers (optimizer list) :   You might want to keep it as empty list and update the list when self.compile function is called. You can define one optimizer for each network here. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan.py for an example.
        """

        self.gpu_ids = gpu_ids

        # get device name: CPU or GPU
        if device == "cuda":
            if torch.cuda.is_available():
                
                if self.gpu_ids == None:
                    self.gpu_ids = [i for i in range(torch.cuda.device_count())]

                self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            
            else:
                print("CUDA not available falling back to CPU.")
                self.device = torch.device("cpu")

        elif device == "cpu":
            self.device = torch.device("cpu")
        
        #  initializing lists
        self.model_names = []
        self.loss_names = []
        self.optimizer_names = []
        self.optimizers = []
        # self.visual_names = []
        # self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.epoch = 0

    @abstractclassmethod
    def from_disc(cls, dir_path:str):
        """
        Loads the models from the disc and initializes the class.
        """
        pass

    @abstractmethod
    def set_data(self, batch: tuple):
        """
        Unpacks a single batch data from the dataloader and perform necessary steps.
        
        :param batch (tuple): single batch of data, (X, y)
        :return: None
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        pass

    @abstractmethod
    def compile(self,):
        """
        Used to set and initialize optimizers, loss, metrics and lr schedulers.
        """
        pass

    @abstractmethod
    def fit(self,):
        """
        Trains the networks
        """
        pass

    # def set_scheduler(self, lr_policy: str, schedulers = [], **kwargs) -> None:
    #     """
    #     Create schedulers
        
    #     :param lr_policy: name of learning rate policy: linear | step | plateau | cosine
    #     :param verbose: verbose for printing the networks, True or False
    #     :param kwargs: keyword arguments to set for learning rate
    #                    linear : epoch_count | n_epochs | n_epochs_decay
    #                    step   : n_decay_iters
    #                    plateau: 
    #                    cosine : n_epochs

    #     :return: None            
    #     """
    #     if len(schedulers) == 0:
    #         self.schedulers = [get_scheduler(optimizer, lr_policy, **kwargs) for optimizer in self.optimizers]
    #     else:
    #         self.schedulers = schedulers

    def set_mode(self, mode) -> None:
        """
        Make models eval mode during test time.
        
        :return: None
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                
                if mode == 'train':
                    net.train()
                elif mode == 'eval':
                    net.eval()
                else:
                    raise Exception(f'Mode: {mode} is not recognized')

    def set_requires_grad(self, nets: list, requires_grad:bool = False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        
        :param nets (network list)  :   a list of networks
        :param requires_grad (bool) :   whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def test(self) -> None:
        """
        Forward function used in test time.
        
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results

        :return: None
        """
        with torch.no_grad():
            self.forward()
            # self.compute_visuals()

    # def update_learning_rate(self) -> None:
    #     """
    #     Update learning rates for all the networks; called at the end of every epoch
    #     """

    #     old_lr = self.optimizers[0].param_groups[0]['lr']
    #     for scheduler in self.schedulers:
    #         if self.lr_policy == 'plateau':
    #             scheduler.step(self.metric) # TODO metric here is validation loss
    #         else:
    #             scheduler.step()

    #     lr = self.optimizers[0].param_groups[0]['lr']
    #     print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self) -> OrderedDict:
        """
        Return traning losses / errors. train.py will print out these errors on console, and save them to a file
        """

        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def print_networks(self, verbose=False) -> None:
        """
        Print the total number of parameters in the network and (if verbose) network architecture
        
        :param verbose (bool): if verbose: print the network architecture
        :return: None
        """

        print('----------------------------------------- Networks -----------------------------------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('-' * 10, f' [Network {name}] Total number of parameters : {format(num_params / 1e6, ".2f")} M ', '-' * 10)
                if verbose:
                    print(net)
        print('--------------------------------------------------------------------------------------------')

    def get_current_checkpoint_name(self) -> str:
        '''
        Created current checkpoint name
        :return (str): checkpoint name without extension
        '''

        current_losses = self.get_current_losses()
        loss_str = '_'.join([str(i) + str(current_losses[i]) for i in current_losses])
        name = f'{self.__class__.__name__}_epoch{self.epoch}_{loss_str}'
        return name

    def save_networks(self, dir_path:str) -> None:
        """
        Save all the networks to the disk.        
        :param dir (str): networks will be saved at given directory path
        :return: None
        """
        dir_path = Path(dir_path)
        dir_path = dir_path / self.get_current_checkpoint_name()
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving Networks at {dir_path}")

        for name in self.model_names:
            if isinstance(name, str):
                file_name = f'net_{name}.pt'
                file_path = dir_path / file_name
                net = getattr(self, 'net_' + name)

                if 'cuda' in str(self.device):
                    torch.save(net.module.cpu(), file_path)
                    net.to(self.device)
                else:
                    torch.save(net, file_path)

    def load_networks(self, dir_path: str) -> None:
        """
        Load all the networks from the disk.        
        :param dir_path: networks will be loaded from given directory path
        :return: None
        """
        dir_path = Path(dir_path)

        for name in self.model_names:
            if isinstance(name, str):
                file_name = f'net_{name}.pt'
                file_path = dir_path / file_name
                print(f'loading the model from {file_path}')        
                net = torch.load(file_path)
                net.to(self.device)
                if 'cuda' in str(self.device) :
                    net = torch.nn.DataParallel(net, self.gpu_ids)
                setattr(self, 'net_' + name, net)

        print("Networks have been loaded, make sure to compile again.")

    def save_checkpoints(self, dir_path: str) -> None:
        '''
        Saves the checkpoint at given diretory path.
        :param dir_path (str): checkpoint will be saved at given file
        :return: None
        '''

        dir_path = Path(dir_path)
        dir_path = dir_path / self.get_current_checkpoint_name()
        dir_path.mkdir(parents=True, exist_ok=True)

        print(f'Saving checkpoint at {dir_path}')

        # For saving models
        for net_name in self.model_names:
            file_path = dir_path / ('net_' + net_name + '.pt')
            checkpoint = {}
            net = getattr(self, 'net_' + net_name)
            
            if isinstance(net, nn.DataParallel):
                state_dict = net.module.cpu().state_dict()
                net.to(self.device)
            else:
                state_dict = net.state_dict()

            checkpoint['net_' + net_name + '_state_dict'] = state_dict
            torch.save(checkpoint, file_path)

        # For saving Optimizers
        optims = {}
        for optim_name in self.optimizer_names:
            optim = getattr(self, 'optimizer_' + optim_name)
            optims['optimizer_' + optim_name + '_state_dict'] = optim.state_dict()
            torch.save(optims, dir_path / 'optimizers.pt')

        # Losses and Epoch will be saved in meta.pt
        # For saving Losses
        meta = {}
        current_losses = self.get_current_losses()
        for loss_name in current_losses:
            meta['loss_' + loss_name] = current_losses[loss_name]

        # For saving epoch
        meta['epoch'] = self.epoch

        # Sving the data
        torch.save(meta, dir_path / 'meta.pt')

    def load_checkpoints(self, dir_path:str):
        '''
        Loads all the model checkpoints from the disk.

        :param dir_path (str): checkpoint will be loaded from given directory
        :return: None
        '''
        dir_path = Path(dir_path)
        print(f'Loading checkpoint from {dir_path}')

        # Loading Models
        for net_name in self.model_names:

            net = getattr(self, 'net_' + net_name)
            file_path = dir_path / ('net_' + net_name + '.pt')
            checkpoint = torch.load(file_path, map_location=str(self.device))
            state_dict = checkpoint['net_' + net_name + '_state_dict']
            if isinstance(net, nn.DataParallel):
                net = net.module            
            net.load_state_dict(state_dict)


        # Loading Optimizers
        optims = torch.load(dir_path / 'optimizers.pt')
        for optim_name in self.optimizer_names:

            optim = getattr(self, 'optimizer_' + optim_name)
            optim.load_state_dict(optims['optimizer_' + optim_name + '_state_dict'])

        # Loading Losses and Epoch from meta.pt
        meta = torch.load(dir_path / 'meta.pt')
        for loss_name in self.loss_names:
            setattr(self, 'loss_' + loss_name, meta['loss_' + loss_name])

        self.epoch = meta['epoch']

    # def compute_visuals(self):
    #     """
    #     Calculate additional output images for visdom and HTML visualization
    #     """
    #     pass

    # def get_image_paths(self):
    #     """ 
    #     Return image paths that are used to load current data
    #     """
    #     return self.image_paths

    # def get_current_visuals(self):
    #     """
    #     Return visualization images. train.py will display these images with visdom, and save the images to a HTML
    #     """
        
    #     visual_ret = OrderedDict()
    #     for name in self.visual_names:
    #         if isinstance(name, str):
    #             visual_ret[name] = getattr(self, name)
    #     return visual_ret
