from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torchvision.datasets as dset
import torch

class CelebA_Data(data.Dataset):
    def __init__(self, config):
        super().__init__()

        self.dataset = dset.ImageFolder(root=config.data_dir,
                           transform=T.Compose([
                               T.Resize(config.image_size),
                               T.CenterCrop(config.image_size),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x,y


def get_loader(batch_size, mode, num_workers, config):
     
    datasets = CelebA_Data(config)
    data_loader = DataLoader(dataset = datasets,
                            batch_size = batch_size,
                            shuffle = (mode=="train"),
                            num_workers = num_workers,
                            drop_last=True)
    return data_loader
        

      
