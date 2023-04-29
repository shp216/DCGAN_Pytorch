from dataloader import get_loader
from solver import Solver
import os
import argparse
from torch.backends import cudnn

def main(config):
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir): 
        os.makedirs(config.result_dir)

    # Solver for training and testing VanillaGAN.
    solver = Solver(config)
    
    if config.mode == "train":
        solver.train(config)
    
    elif config.mode == "test":
        solver.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()        

    # Model Configuration
    parser.add_argument('--image_size', type=int, default=64, help='image_size')
    parser.add_argument('--latent_size', type=int, default=100, help='size of latent_vector used in Generator')

    # Miscellaneous Configuration
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'])
    parser.add_argument('--num_workers', type=int, default=0)

    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default="data/celeba")

    #Training Configuration
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch_size')
    parser.add_argument('--num_epochs', type=int, default=5, help='total epochs')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images. For color images this is 3')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector (i.e. size of generator input')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparam for Adam optimizers')

    config = parser.parse_args()
    print(config)
    main(config)
    
