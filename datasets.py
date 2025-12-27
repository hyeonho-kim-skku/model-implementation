import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

class SimCLRAugmentation():
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), # kernel size is set to be 10% of the image height/width.
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
    
    def __call__(self, x):
        xi = self.train_transform(x)
        xj = self.train_transform(x)
        return xi, xj

class CIFAR10Pair(Dataset):
    def __init__(self):
        self.base_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.transform = SimCLRAugmentation()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        xi, xj = self.transform(img)
        return xi, xj

def load_dataset(dataset_name, batch_size):
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) # mean, std
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) # mean, std
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
    elif dataset_name == 'CIFAR10_SimCLR':
        trainset = CIFAR10Pair()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

        return trainloader, None