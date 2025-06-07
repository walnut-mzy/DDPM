import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os


class MNISTDataset:

    def __init__(
        self,
        root_dir="./data",
        image_size=32,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        download=True,
        use_augmentation=False,
        normalize_to_neg_one_to_one=True,
        convert_to_rgb=True
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.use_augmentation = use_augmentation
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.convert_to_rgb = convert_to_rgb
        
        os.makedirs(root_dir, exist_ok=True)
        
        self.setup_transforms()
        
        self.load_datasets()
        
        self.create_dataloaders()
        
        channels = 3 if self.convert_to_rgb else 1
        print(f"MNIST数据集加载完成:")
        print(f"  - 训练集大小: {len(self.train_dataset)}")
        print(f"  - 测试集大小: {len(self.test_dataset)}")
        print(f"  - 图像尺寸: {self.image_size}x{self.image_size}")
        print(f"  - 图像通道数: {channels} ({'RGB' if self.convert_to_rgb else '灰度'})")
        print(f"  - 批大小: {self.batch_size}")
        
    def setup_transforms(self):
        transform_list = []
        
        if self.image_size != 28:
            transform_list.append(transforms.Resize(self.image_size))
        
        transform_list.append(transforms.ToTensor())
        
        if self.convert_to_rgb:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        if self.normalize_to_neg_one_to_one:
            if self.convert_to_rgb:
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            if self.convert_to_rgb:
                transform_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
            else:
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        self.base_transform = transforms.Compose(transform_list)
        
        if self.use_augmentation:
            augment_list = [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ]
            
            if self.image_size != 28:
                augment_list.append(transforms.Resize(self.image_size))
            
            augment_list.append(transforms.ToTensor())
            
            if self.convert_to_rgb:
                augment_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            
            if self.normalize_to_neg_one_to_one:
                if self.convert_to_rgb:
                    augment_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                else:
                    augment_list.append(transforms.Normalize((0.5,), (0.5,)))
            else:
                if self.convert_to_rgb:
                    augment_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
                else:
                    augment_list.append(transforms.Normalize((0.1307,), (0.3081,)))
            
            self.train_transform = transforms.Compose(augment_list)
        else:
            self.train_transform = self.base_transform
        
        self.test_transform = self.base_transform
    
    def load_datasets(self):
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=True,
            transform=self.train_transform,
            download=self.download
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=False,
            transform=self.test_transform,
            download=self.download
        )
    
    def create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_sample_batch(self, from_train=True):
        loader = self.train_loader if from_train else self.test_loader
        return next(iter(loader))
    
    def denormalize(self, tensor):
        if self.normalize_to_neg_one_to_one:
            return (tensor + 1.0) / 2.0
        else:
            if self.convert_to_rgb:
                mean = torch.tensor([0.1307, 0.1307, 0.1307]).view(1, 3, 1, 1)
                std = torch.tensor([0.3081, 0.3081, 0.3081]).view(1, 3, 1, 1)
                if tensor.device != mean.device:
                    mean = mean.to(tensor.device)
                    std = std.to(tensor.device)
                return tensor * std + mean
            else:
                return tensor * 0.3081 + 0.1307
    
    def convert_to_rgb(self, images):
        if self.convert_to_rgb:
            return images
        else:
            return images.repeat(1, 3, 1, 1)
    
    def get_class_names(self):
        return [str(i) for i in range(10)]
    
    def get_dataset_info(self):
        channels = 3 if self.convert_to_rgb else 1
        return {
            "num_classes": 10,
            "num_channels": channels,
            "image_size": self.image_size,
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "class_names": self.get_class_names(),
            "normalization": "[-1, 1]" if self.normalize_to_neg_one_to_one else "[0, 1]",
            "convert_to_rgb": self.convert_to_rgb
        }
    
    def visualize_samples(self, num_samples=8, from_train=True, save_path=None):
        import matplotlib.pyplot as plt
        
        images, labels = self.get_sample_batch(from_train)
        images = self.denormalize(images)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            if self.convert_to_rgb:
                img = images[i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
            else:
                img = images[i].squeeze().numpy()
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"样本可视化已保存到: {save_path}")
        else:
            plt.show()


