from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

CONFIG = {
    'cifar10': {
        'mean' : (0.4914, 0.4822, 0.4465),
        'std' : (0.2470, 0.2435, 0.2616),
        'size' : 32,
        'class' : datasets.CIFAR10,
        'ssl_params' : {
            'blur_prob' : 0.0 # CIFAR-10은 blur 적용하지 않음.
        }
    }
}

class TwoCropTransform:
    """1개의 image를 2개의 view로 변환"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        v1 = self.transform(x)
        v2 = self.transform(x)
        return [v1, v2]

def get_transform(dataset_name, mode):
    """
    dataset_name (str): 'cifar10' or ...
    mode (str): 'supervised', 'ssl', 'test'
    """
    config = CONFIG[dataset_name]
    img_size = config['size']

    # 공통 후처리 (모든 모드 공통)
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ]

    # mode 별 Augmentation 정의 및 반환
    if mode == 'supervised':
        augmentations = [
            transforms.RandomResizedCrop(img_size), # default scale = (0.08, 1.0)
            transforms.RandomHorizontalFlip() # default p=0.5
        ]
        return transforms.Compose(augmentations + common_transforms)
    elif mode == 'ssl':
        augmentations = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]
        # blur 추가 (kernel size = image size의 10% 홀수)
        blur_prob = CONFIG[dataset_name]['ssl_params']['blur_prob']
        if blur_prob > 0:
            kernel_size = int(img_size * 0.1)
            if kernel_size % 2 == 0: kernel_size += 1
            augmentations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=blur_prob))
        return TwoCropTransform(transforms.Compose(augmentations + common_transforms))
    elif mode == 'test':
        return transforms.Compose(common_transforms)
    
def get_loader(dataset_name, batch_size, mode, train, shuffle, drop_last, num_workers=4, data_root='./data'):
    # Transform 생성
    transform = get_transform(dataset_name, mode)

    # DataSet 생성
    dataset_class = CONFIG[dataset_name]['class']
    dataset = dataset_class(root=data_root, train=train, download=True, transform=transform)

    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return dataloader
