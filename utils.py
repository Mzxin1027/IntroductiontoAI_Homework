import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math

class LabelSmoothingCrossEntropy(nn.Module):
   
    def __init__(self, alpha=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        true_dist = torch.full_like(log_probs, self.alpha / (num_classes -1 if num_classes > 1 else 1) )
       
        if targets.ndim == 1:
            targets_expanded = targets.unsqueeze(1)
        else:
            targets_expanded = targets
        true_dist.scatter_(1, targets_expanded, 1.0 - self.alpha)
        
        loss = -torch.sum(true_dist * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CosineAnnealingLRWithWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=5, warmup_lr_init=1e-6, last_epoch=-1):
        self.T_max = T_max - warmup_epochs
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.T_cur = 0
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr_ratio = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_lr_init + lr_ratio * (base_lr - self.warmup_lr_init) for base_lr in self.base_lrs]
        else:
            self.T_cur = self.last_epoch - self.warmup_epochs 
            return [self.eta_min + (base_lr - self.eta_min) * \
                    (1 + math.cos(math.pi * self.T_cur / self.T_max)) / 2
                    for base_lr in self.base_lrs]

def get_cifar10_loaders(batch_size=128, data_dir='./data', num_workers=4, use_augmentation=True, image_size=32):
  
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    print("Testing LabelSmoothingCrossEntropy...")
    criterion_ls = LabelSmoothingCrossEntropy(alpha=0.1)
    example_logits = torch.randn(2, 5)
    example_targets = torch.tensor([0, 2])
    loss_ls = criterion_ls(example_logits, example_targets)
    print(f"Label smoothing loss: {loss_ls.item()}")

    print("\nTesting CosineAnnealingLRWithWarmup...")
    model_param = nn.Linear(10,1)
    optimizer_test = optim.Adam(model_param.parameters(), lr=0.001)
    scheduler_test = CosineAnnealingLRWithWarmup(optimizer_test, T_max=150, warmup_epochs=10, eta_min=1e-5, warmup_lr_init=1e-7)
    
    print("LR schedule:")
    for epoch in range(150):
        scheduler_test.step()
        if epoch < 15 or epoch > 140 or epoch % 10 == 0:
             print(f"Epoch {epoch+1}: LR = {scheduler_test.get_lr()[0]:.8f}")

    print("\nTesting CIFAR-10 data loaders...")
    try:
        train_loader_test, test_loader_test = get_cifar10_loaders(batch_size=4, image_size=32, num_workers=0)
        print(f"Number of training batches: {len(train_loader_test)}")
        print(f"Number of test batches: {len(test_loader_test)}")
        images, labels = next(iter(train_loader_test))
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading data: {e}. CIFAR-10 might not be downloaded. You may need to run this part manually once.")

    print("\nTesting CIFAR-10 data loaders with 224x224 resize...")
    try:
        train_loader_224, _ = get_cifar10_loaders(batch_size=4, image_size=224, num_workers=0)
        images_224, _ = next(iter(train_loader_224))
        print(f"Images batch shape (224x224): {images_224.shape}")
    except Exception as e:
        print(f"Error loading data (224x224): {e}.")