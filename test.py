import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#grabbing the datasets
#get the training data with labels:
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    )

#get the test data withOUT labels:
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    )
    
print (training_data)
print ("=============================")
print (test_data)

#specify the number of features/labels each element of the dataloader will return
batch_size = 64

#make a dataloader for both the training and test datasets
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

'''
#will just return the object memloc
print (train_dataloader)
print ("=============================")
print (test_dataloader)
'''
#.shape gives the shape of the torchvision tensor
#that is to say, that a tensor is an n-dimensional array (4 in the case of feature_batch, 1 in the case 
#of label_batch)
#[N, C, H, W] corresponds to [batch_size, number of color channels, height(pixels), width(pixels)
for feature_batch, label_batch in test_dataloader:
    print(f"Shape of feature_batch [N, C, H, W]: {feature_batch.shape}")
    print(f"Shape of label_batch: {label_batch.shape} {label_batch.dtype}")
    break