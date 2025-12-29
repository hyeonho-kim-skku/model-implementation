import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

def get_base_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_base_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_base_test_loader(batch_size):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_base_test_transform())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

class SimCLRAugmentation():
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],p=0.5), # # kernel size is set to be 10% of the image height/width.
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    def __call__(self, x):
        xi = self.train_transform(x)
        xj = self.train_transform(x)
        return xi, xj
    
class MoCoAugmentation():
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],p=0.5), # v2는 blur 추가.
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

    def __call__(self, x):
        query = self.moco_transform(x)
        key = self.moco_transform(x)
        return query, key

class CIFAR10SimCLR(Dataset):
    def __init__(self):
        self.base_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.transform = SimCLRAugmentation()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        xi, xj = self.transform(img)
        return (xi, xj), label

class CIFAR10MoCo(Dataset):
    def __init__(self):
        self.base_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.transform = MoCoAugmentation()

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        query, key = self.transform(img)
        return (query, key), label

def load_dataset(dataset_name, batch_size):
    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_base_train_transform())
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        return trainloader, get_base_test_loader(batch_size)
    elif dataset_name == 'CIFAR10_SimCLR':
        trainset = CIFAR10SimCLR()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        return trainloader, get_base_test_loader(batch_size)
    elif dataset_name == 'CIFAR10_MoCo':
        trainset = CIFAR10MoCo()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        return trainloader, get_base_test_loader(batch_size)
    elif dataset_name == 'knn_train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_base_test_transform())
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        return trainloader
    

