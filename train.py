import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import os
import time
import argparse
import numpy as np

from model import ImprovedResNet50, ImprovedResNet18
from utils import LabelSmoothingCrossEntropy, CosineAnnealingLRWithWarmup, get_cifar10_loaders


CONFIG = {
    'dataset': 'cifar10',
    'num_classes': 10,
    'model_name': 'ImprovedResNet50', 
    'use_se': True, 
    'image_size': 32,
    
    'batch_size': 128, 
    'num_epochs': 30, 
    'learning_rate': 0.001, 
    'optimizer': 'adam', 
    'weight_decay': 1e-4, 
    'label_smoothing_alpha': 0.1, 
    'lr_scheduler': 'cosine_annealing_warmup',
    'warmup_epochs': 10, 
    'eta_min': 1e-5, 
    
    'use_mixed_precision': False, 
    'gradient_clipping_norm': 1.0, 
    
    'data_dir': './data',
    'save_dir': './checkpoints',
    'log_interval': 50, 
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'resume_checkpoint': './checkpoints/last_checkpoint.pth' 
}

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, config):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config['device']), target.to(config['device'])

        optimizer.zero_grad()

        if config['use_mixed_precision']:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            if config['gradient_clipping_norm'] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if config['gradient_clipping_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping_norm'])
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        if batch_idx % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            print(f'Epoch: {epoch+1}/{config["num_epochs"]} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}s')
            start_time = time.time()
            
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, config):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            
            if config['use_mixed_precision']:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct_predictions / total_samples
   
    return avg_loss, accuracy

def main(config):
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

    print(f"Loading {config['dataset']} dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'], 
        data_dir=config['data_dir'], 
        num_workers=config['num_workers'],
        image_size=config['image_size'] 
    )
    print(f"Data loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print(f"Initializing model: {config['model_name']} with SE: {config['use_se']}")
    if config['model_name'] == 'ImprovedResNet50':
        model = ImprovedResNet50(num_classes=config['num_classes'], use_se=config['use_se'])
    elif config['model_name'] == 'ImprovedResNet18':
        model = ImprovedResNet18(num_classes=config['num_classes'], use_se=config['use_se'])
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")
    model.to(config['device'])

    if config['label_smoothing_alpha'] > 0:
        criterion = LabelSmoothingCrossEntropy(alpha=config['label_smoothing_alpha'])
    else:
        criterion = nn.CrossEntropyLoss()
    print(f"Using loss: {'LabelSmoothingCrossEntropy' if config['label_smoothing_alpha'] > 0 else 'CrossEntropyLoss'}")

    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    print(f"Using optimizer: {config['optimizer']}")

    scheduler = None
    if config['lr_scheduler'] == 'cosine_annealing_warmup':
        scheduler = CosineAnnealingLRWithWarmup(
            optimizer, 
            T_max=config['num_epochs'], 
            eta_min=config['eta_min'], 
            warmup_epochs=config['warmup_epochs']
        )
        print(f"Using LR scheduler: CosineAnnealingLRWithWarmup")
    elif config['lr_scheduler'] == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        print(f"Using LR scheduler: StepLR")

    scaler = GradScaler(enabled=config['use_mixed_precision'])
    print(f"Using mixed precision: {config['use_mixed_precision']}")

    start_epoch = 0
    best_val_accuracy = 0.0
    if config['resume_checkpoint'] and os.path.exists(config['resume_checkpoint']):
        print(f"Resuming from checkpoint: {config['resume_checkpoint']}")
        checkpoint = torch.load(config['resume_checkpoint'], map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        if config['use_mixed_precision'] and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed from epoch {start_epoch}, best val accuracy: {best_val_accuracy:.2f}%")

    print("Starting training...")
    os.makedirs(config['save_dir'], exist_ok=True)

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, config)
        val_loss, val_acc = evaluate(model, test_loader, criterion, config)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch: {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s")

        if scheduler:
            scheduler.step()

        is_best = val_acc > best_val_accuracy
        best_val_accuracy = max(val_acc, best_val_accuracy)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'config': config
        }
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        if config['use_mixed_precision']:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_data, os.path.join(config['save_dir'], 'last_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint_data, os.path.join(config['save_dir'], 'best_checkpoint.pth'))
            print(f"Saved new best model with accuracy: {best_val_accuracy:.2f}% at epoch {epoch+1}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training based on Report')
    for key, value in CONFIG.items():
        arg_type = type(value)
        if arg_type == bool:
            parser.add_argument(f'--{key.replace("_", "-")}', action='store_true', default=value)
            parser.add_argument(f'--no-{key.replace("_", "-")}', action='store_false', dest=key)
        else:
            parser.add_argument(f'--{key.replace("_", "-")}', type=arg_type, default=value)
    
    args = parser.parse_args()
    config_updates = {k: v for k, v in vars(args).items() if v is not None}
    CONFIG.update(config_updates)

    print("Current Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    main(CONFIG)