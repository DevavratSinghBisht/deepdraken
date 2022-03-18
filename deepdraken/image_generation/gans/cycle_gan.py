'''
Reference Code at 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9e6fff7b7d5215a38be3cac074ca7087041bea0d/models/cycle_gan_model.py#L8
'''

from multiprocessing import pool
import torch
import itertools
from deepdraken.utils.base_utils import ImagePool
from deepdraken.base import BaseModel


class CycleGAN(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--net_G resnet_9blocks' ResNet generator,
    a '--net_D basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        
        Returns:
            the modified parser.
        
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self,
                 net_G_A,
                 net_G_B,
                 net_D_A,
                 net_D_B,
                 pool_size,
                 device,
                 is_train = True,
                 gpu_ids = []):
        
        """
        Initialize the CycleGAN class.
        
        :param net_G_A: 
        """

        super().__init__(device, is_train, gpu_ids)

        # Networks
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.net_G_A = net_G_A
        self.net_G_B = net_G_B
        self.net_D_A = net_D_A
        self.net_D_B = net_D_B

        self.is_train = is_train

        # specify the models you want to save to the disk. 
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # specify the training losses you want to print out. 
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        self.optimizer_names = ['G', 'D']
        
        # specify the images you want to save/display. 
        # The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_B']
        self.visual_names = visual_names_A + visual_names_B

        self.pool_size = pool_size
        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        # define loss functions
        self.criterionGAN = torch.nn.MSELoss.to(self.device)
        self.criterionCycle = torch.nn.L1Loss().to(self.device)
        self.criterionIdt = torch.nn.L1Loss().to(self.device)

    def set_data(self, batch_A, batch_B):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = batch_A[0].to(self.device)
        self.real_B = batch_B[0].to(self.device)

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """

        self.fake_B = self.net_G_A(self.real_A)  # G_A(A)
        self.rec_A = self.net_G_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.net_G_B(self.real_B)  # G_B(B)
        self.rec_B = self.net_G_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, net_D, real, fake):
        """
        Calculate GAN loss for the discriminator.
        We also call loss_D.backward() to calculate the gradients.
        
        :param net_D (network): the discriminator D
        :param real (tensor array): real images
        :param fake (tensor array): images generated by a generator
        
        :return: the discriminator loss.
        """
        # Real
        pred_real = net_D(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = net_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """
        Calculate GAN loss for discriminator D_A
        """
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.net_D_A, self.real_B, fake_B)

    def backward_D_B(self):
        """
        Calculate GAN loss for discriminator D_B
        """

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.net_D_B, self.real_A, fake_A)

    def backward_G(self):
        """
        Calculate the loss for generators G_A and G_B
        """

        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.net_D_A(self.fake_B), True)  # D_A(G_A(A)) 
        self.loss_G_B = self.criterionGAN(self.net_D_B(self.fake_A), True)  # D_B(G_B(B))
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.net_G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.net_G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """

        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.net_D_A, self.net_D_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.net_D_A, self.net_D_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def compile(self,
                optim_G,
                optim_D,
                optim_params_G: dict = {},
                optim_params_D: dict = {},
                lambda_identity: float = 0.5,
                lambda_A: float = 10.0,
                lambda_B: float = 10.0):
        
        self.optimizer_G = optim_G(itertools.chain(self.net_G_A.parameters(), self.net_G_B.parameters()), **optim_params_G)
        self.optimizer_D = optim_D(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()), **optim_params_D)
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        if self.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            # TODO condition to check if nets have same number of input and output channels
            # assert(opt.input_nc == opt.output_nc)
            pass

    def fit(self,
            data_A,
            data_B,
            epochs, 
            batch_size = 64, 
            shuffle = True, 
            transform = None):
        
        # Setting nets to training mode
        self.set_mode('train')

        def get_dataloader_if_needed(data):
            if isinstance(data, str):
                data = self.get_dataloader(data, batch_size, shuffle, transform)
            elif isinstance(data, torch.Tensor):
                pass # TODO for direct tensor inputs
            return data

        data_A = get_dataloader_if_needed(data_A)
        data_B = get_dataloader_if_needed(data_B)

        for epoch in range(epochs): # TODO use TQDM here
            for batch_idx, (batch_A, batch_B) in enumerate(zip(data_A, data_B)):

                self.set_data(batch_A, batch_B)
                self.optimize_parameters()

            # TODO print the mean loss of whole epoch rather than for the latest batch
            print("[Epoch {}/{}] [G loss: {}] [D_A loss: {}] [D_B loss {}]".format(epoch, epochs, self.loss_D, self.loss_G))

# TODO confirm loss calculations
# TODO we are using two dataloader here, change that to one