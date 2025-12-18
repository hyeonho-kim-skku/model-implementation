import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data.dataloader import default_collate

def rotate_img(img, rot):
    img = np.array(img, copy=True)
    
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif rot == 180: # 180 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

class RotationDataset(Dataset):
    """비지도 학습을 위한 래퍼 데이터셋 - 한 이미지에서 4가지 회전 버전 생성"""
    def __init__(self, dataset, epoch_size):
        self.dataset = dataset # 원본 데이터셋
        self.epoch_size = epoch_size # 한 에폭에서 볼 샘플 수

    def __len__(self):
        return self.epoch_size # 에폭 크기만큼 반환

    def __getitem__(self, index):
        # 데이터셋 크기를 넘으면 처음부터 다시(무한 반복)
        idx = index % len(self.dataset)
        img0, _ = self.dataset[idx] # 원본 이미지 가져오기 (라벨은 무시) 뒤에서 회전각도(0,1,2,3) 생성.

        # 같은 이미지의 4가지 버전 생성 + 정규화 적용
        rotated_imgs = [
            self.dataset.transform(rotate_img(img0, 0)), # 0도
            self.dataset.transform(rotate_img(img0, 90)), # 90도
            self.dataset.transform(rotate_img(img0, 180)), # 180도
            self.dataset.transform(rotate_img(img0, 270)) # 270도
        ]
        # 회전 각도를 라벨로 사용 (0, 1, 2, 3)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])

        # (4, C, H, W) 형태의 텐서와 (4,) 라벨 반환
        return torch.stack(rotated_imgs, dim=0), rotation_labels

class RotNetDataLoader():
    def __init__(self, dataset, batch_size, unsupervised, epoch_size, shuffle, num_workers=2):
        self.dataset = dataset # CIFAR10RotNetDataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset) # 한 에폭 샘플 수 (None이면 데이터셋 전체)
        self.batch_size = batch_size
        self.unsupervised = unsupervised # 비지도 학습 여부 (회전 예측)
        self.num_workers = num_workers
        
        self._current_epoch = 0

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])

        # dataset의 transform 업데이트.
        self.dataset.transform = self.transform
    
    def __iter__(self):
        return iter(self.get_iterator())  # 내부 DataLoader 반환

    def get_iterator(self):
        if self.unsupervised:
            # 비지도 학습: RotationDataset 래퍼 사용.
            wrapped_dataset = RotationDataset(self.dataset, self.epoch_size)
            data_loader = DataLoader(wrapped_dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers, 
                                     collate_fn=self._collate_unsupervised) # 배치 포맷 변경 함수
        else:
            pass
            # 지도 학습: 원본 데이터셋 그대로 사용 (기존 cifar10 데이터 사용.)
            # data_loader = DataLoader(
            #     self.dataset,
            #     batch_size=self.batch_size,
            #     shuffle=self.shuffle,
            #     num_workers=self.num_workers
            # )
        return data_loader
    
    def _collate_unsupervised(self, batch): # batch shape: [((4,C,H,W), (4,)) * babtch_size]
        """비지도 학습 배치 포맷 변경 함수 (B,4,C,H,W) -> (B*4,C,H,W)"""
        # 기본 collate로 먼저 묶음: 이미지는 이미지별로 label은 label별로 배치화됨.
        batch = default_collate(batch)
        # (batch_size, 4, C, H, W) 형태 확인
        batch_size, rotations, channels, height, width = batch[0].size()
        # 평탄화
        batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
        batch[1] = batch[1].view([batch_size*rotations])
        return batch

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        """총 배치 수 반환"""
        return self.epoch_size // self.batch_size

"""https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py"""
class CIFAR10RotNetDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]] # [0.4914, 0.4822, 0.4465]
        self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]] # [0.2470, 0.2435, 0.2616]

        transform=[]
        if (split != 'test'):
            transform.append(transforms.RandomCrop(32, padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        # transform.append(lambda x: np.asarray(x))
        self.transform = transforms.Compose(transform)
        self.data = datasets.CIFAR10(
                './data', train=(self.split=='train'),
                download=True, transform=self.transform)
    
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label
    
    def __len__(self):
        return len(self.data)

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
    elif dataset_name == 'CIFAR10_rotnet':
        dataset_train = CIFAR10RotNetDataset(
            split='train'
        )
        dataset_test = CIFAR10RotNetDataset(
            split='test'
        )

        dloader_train = RotNetDataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            unsupervised=True,
            epoch_size=None,
            num_workers=2,
            shuffle=True)
        dloader_test = RotNetDataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            unsupervised=True,
            epoch_size=None,
            num_workers=2,
            shuffle=False)

        return dloader_train, dloader_test
    elif dataset_name == 'CIFAR10_SimCLR':
        trainset = CIFAR10Pair()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        return trainloader, None