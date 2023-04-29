from model import Generator
from model import Discriminator
from dataloader import get_loader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt

class Solver(object):
    
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.latent_size = config.latent_size
        self.image_size = config.image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.model_save_dir = config.model_save_dir
        self.mode = config.mode
        self.num_workers = config.num_workers
        self.num_epochs = config.num_epochs
        self.sample_dir = config.sample_dir
        self.result_dir = config.result_dir
        self.data_dir = config.data_dir
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.beta1 = config.beta1

        self.build_model(config)

    def build_model(self, config):
        """Create a generator and a discriminator."""
        self.G = Generator(config)
        self.D = Discriminator(config)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, betas=(config.beta1, 0.999))
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, betas=(config.beta1, 0.999))

        self.criterion = nn.BCELoss()
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step')
        
        G_path = os.path.join(self.model_save_dir, 'G.ckpt')
        D_path = os.path.join(self.model_save_dir, 'D.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, config):
        data_loader = get_loader(self.batch_size, self.mode, self.num_workers, config)
        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0

        # Weight Initialization
        self.D.apply(self.weights_init)
        self.G.apply(self.weights_init)
        # self.weights_init(self.D)
        # self.weights_init(self.G)
        
        # Start training.
        print('Start training...')
        start_time = time.time()
        total_step = len(data_loader)

        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(data_loader):
                #images = images.reshape(self.batch_size, -1).to(self.device) # (batch_size, 1, 28, 28) -> (batch, 28*28)
                images = images.to(self.device)
                real_label = 1.
                fake_label = 0.
                real_labels = torch.full((self.batch_size,), real_label, dtype=torch.float, device=self.device)
                fake_labels = torch.full((self.batch_size,), fake_label, dtype=torch.float, device=self.device)
                #real_labels = torch.ones(self.batch_size).to(self.device)
                #fake_labels = torch.zeros(self.batch_size).to(self.device)
                
                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #
                
                outputs = self.D(images).view(-1)
                d_loss_real = self.criterion(outputs, real_labels)
                d_real_loss = d_loss_real

                z = torch.randn(self.batch_size, self.latent_size, 1, 1).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)
                
                d_loss_fake = self.criterion(outputs, fake_labels)
                d_fake_loss = d_loss_fake

                d_loss = d_real_loss + d_fake_loss
                self.reset_grad()
                d_loss.backward()
                self.D_optimizer.step()

                # ================================================================== #
                #                        Train the generator                         #
                # ================================================================== #

                z = torch.randn(self.batch_size, self.latent_size, 1, 1).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images).view(-1)
                
                g_loss = self.criterion(outputs, real_labels)

                self.reset_grad()
                g_loss.backward()
                self.G_optimizer.step()


                # =================================================================== #
                #                            Miscellaneous                            #
                # =================================================================== #
                if(epoch == 0 and i == 0):
                    print("Device:", self.device)
                if (i+1) % 50 == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('Elapsed [{}], Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                        .format(et, epoch, self.num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                                d_loss_real.mean().item(), d_loss_fake.mean().item()))

            if (epoch+1) == 1:
                    #images = images.reshape(images.size(0), 3, 128, 128)
                    save_image(self.denorm(images), os.path.join(self.sample_dir, 'real_images.png'))
            
            # Save sampled images
            
            #fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(self.denorm(fake_images), os.path.join(self.sample_dir, 'fake_images-{}.png'.format(epoch+1)))

        # Save the model checkpoints 
        torch.save(self.G.state_dict(), 'G.ckpt')
        torch.save(self.D.state_dict(), 'D.ckpt')
    
    def test(self):
        self.restore_model()
        with torch.no_grad():
            z = torch.randn(10, self.latent_size, 1, 1).to(self.device)
            generated_img = self.G(z)
            save_image(self.denorm(generated_img), os.path.join(self.result_dir, 'Generated_img.png'))
            print('Successfully save Generated images into \'{}\' file !!'.format(self.result_dir))
            # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20,2.5))

            # for i, ax in enumerate(axes):
            #     axes[i].imshow(image[i].to(torch.device("cpu")).detach(), cmap = 'binary')
        
    
        