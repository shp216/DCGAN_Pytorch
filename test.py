# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torch.utils.data

# dataroot = "data/celebA"
# image_size = 128
# batch_size = 128
# workers = 2
# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=workers)

# # Plot some training images
# img, c = next(next(iter(dataloader)))



# print(img.shape)
# print(c)

print("Hello")