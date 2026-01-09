#this is for downloading and preparing datasets
import os
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

root = "./data"

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
])

# training set 经过了增强 填充裁切 和左右反转

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

print(len(train_set), len(test_set))
