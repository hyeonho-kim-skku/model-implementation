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
    },
    'flowers102': {
        'mean' : (0.485, 0.456, 0.406), # ImageNet과 동일한 mean
        'std' : (0.229, 0.224, 0.225), # std 사용
        'size' : 224,
        'class' : datasets.Flowers102,
        'ssl_params' : {
            'blur_prob' : 0.0 # Flower-102는 blur 적용 (논문에서 blur가 성능 향상에 도움된다고 보고됨)
        }
    },
    'fgvc_aircraft': {
        'mean' : (0.485, 0.456, 0.406), # ImageNet pretrained timm 모델과 맞춤
        'std' : (0.229, 0.224, 0.225),
        'size' : 224,
        'class' : datasets.FGVCAircraft,
        'ssl_params' : {
            'blur_prob' : 0.0
        }
    },
    'cifar100': {
        'mean' : (0.485, 0.456, 0.406), # DINOv2 사전 학습 가중치와 맞추기 위해 ImageNet 통계값 사용
        'std' : (0.229, 0.224, 0.225),
        'size' : 224, # DINOv2 입력을 위해 32x32 -> 224x224로 업샘플링
        'class' : datasets.CIFAR100,
        'ssl_params' : {
            'blur_prob' : 0.0 # 작은 이미지를 억지로 키운 것이라 blur는 주지 않는 것이 좋음
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

class MultiCropTransform:
    """1개의 image를 2개의 global crop과 여러 개의 local crop으로 변환"""
    def __init__(self, global_transform, local_transform, local_crops_number=8):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.local_crops_number = local_crops_number
    
    def __call__(self, x):
        crops = []
        # Two global crops - 32x32
        crops.append(self.global_transform(x))
        crops.append(self.global_transform(x))
        
        # Multiple local crops - 16x16
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(x))
        return crops

def get_transform(dataset_name, mode):
    """
    dataset_name (str): 'cifar10' or ...
    mode (str): 'supervised', 'two_crop', 'multi_crop', 'test'
    """
    config = CONFIG[dataset_name]
    img_size = config['size']

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ]

    # mode 별 Augmentation 정의 및 반환
    if mode == 'supervised':
        augmentations = [
            transforms.RandomResizedCrop(img_size), # default scale = (0.08, 1.0)
            transforms.RandomHorizontalFlip() # default p=0.5
        ]
        return transforms.Compose(augmentations + normalize)
    elif mode == 'two_crop':
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
        return TwoCropTransform(transforms.Compose(augmentations + normalize))
    elif mode == 'multi_crop':
        flip_color_gray = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

        global_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_color_gray,
        ] + normalize)

        local_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(img_size * 0.5), scale=(0.2, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_color_gray,
        ] + normalize)

        return MultiCropTransform(global_transform, local_transform, local_crops_number=4)
    elif mode == 'test':
        augmentations = [
            transforms.Resize(256),
            transforms.CenterCrop(img_size)
        ]
        return transforms.Compose(augmentations + normalize)

def get_loader(dataset_name, batch_size, mode, train, shuffle, drop_last, num_workers=4, data_root='./data'):
    # Transform 생성
    transform = get_transform(dataset_name, mode)

    # DataSet 생성
    dataset_class = CONFIG[dataset_name]['class']
    if dataset_name == 'flowers102':
        split = 'train' if train else 'test'
        dataset = dataset_class(root=data_root, split=split, download=True, transform=transform)
    elif dataset_name == 'fgvc_aircraft':
        # Use train+val for training, which is the common FGVC-Aircraft protocol.
        split = 'trainval' if train else 'test'
        dataset = dataset_class(root=data_root, split=split, download=True, transform=transform)
    else:
        dataset = dataset_class(root=data_root, train=train, download=True, transform=transform)

    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return dataloader
