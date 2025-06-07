import torch
import torch.nn as nn
from diffusion import create_unet, DiffusionModel
from dataprocess import MNISTDataset
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np

def print_training_parameters(config):
    print("\n" + "="*50)
    print("训练配置参数")
    print("="*50)
    
    print(f"\n[基础参数]")
    print(f"设备: {config['device']}")
    print(f"批大小: {config['batch_size']}")
    print(f"总训练轮次: {config['epochs']}")
    print(f"保存模型路径: {config['save_dir']}")
    
    print(f"\n[优化器参数]")
    print(f"优化器类型: {config['optimizer_type']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"权重衰减: {config['weight_decay']}")
    print(f"使用学习率调度器: {config['use_scheduler']}")
    if config['use_scheduler']:
        print(f"调度器类型: {config['scheduler_type']}")
    
    print(f"\n[模型参数]")
    print(f"模型类型: UNet + Diffusion")
    print(f"总参数量: {config['total_params']:,}")
    print(f"图像尺寸: {config['image_size']}x{config['image_size']}")
    print(f"图像通道数: {config['channels']}")
    print(f"扩散步数: {config['num_timesteps']}")
    print(f"基础通道数: {config['base_channels']}")
    print(f"残差块数: {config['num_res_blocks']}")
    print(f"通道倍数: {config['channel_mult']}")
    print(f"Dropout率: {config['dropout']}")
    print(f"时间嵌入维度: {config['time_emb_dim']}")
    print(f"条件生成: {'启用' if config['use_conditional_generation'] else '禁用'}")
    if config['use_conditional_generation']:
        print(f"类别数量: {config['num_classes']}")
    
    print(f"\n[EMA参数]")
    print(f"EMA衰减率: {config['ema_decay']}")
    print(f"EMA开始步数: {config['ema_start']}")
    print(f"EMA更新频率: {config['ema_update_rate']}")
    
    print(f"\n[数据集参数]")
    print(f"数据集: MNIST")
    print(f"训练集大小: {config['train_size']}")
    print(f"测试集大小: {config['test_size']}")
    print(f"数据增强: {config['use_augmentation']}")
    print(f"归一化范围: {config['normalization']}")
    print(f"转换为RGB: {config.get('convert_to_rgb', False)}")
    print("="*50 + "\n")
    
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)
 
        loss = model(images, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.update_ema()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    
    if writer is not None:
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        writer.add_scalar('Time/Epoch_Duration', elapsed, epoch)
    
    print(f"Epoch {epoch+1} - 训练损失: {avg_loss:.4f}, 用时: {elapsed:.2f}秒")
    
    return avg_loss

def evaluate(model, dataloader, device, epoch=None, writer=None):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            loss = model(images, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
    
    print(f"评估 - 平均损失: {avg_loss:.4f}")
    
    return avg_loss

def save_model(model, optimizer, epoch, loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")

def generate_and_log_samples(model, dataset, device, epoch, writer, save_dir, 
                           batch_size=5, channels=1, image_size=28):
    print(f"生成第{epoch+1}轮的样本图像...")
    model.eval()
    
    with torch.no_grad():
        samples = model.sample_diffusion_sequence(
            batch_size=batch_size,
            device=device
        )
    
    generated_images = samples[-1]
    
    denorm_images = dataset.denormalize(generated_images)
    
    grid = vutils.make_grid(denorm_images, nrow=5, padding=2, normalize=False)
    
    writer.add_image(f'Generated_Images/Epoch_{epoch+1}', grid, epoch)
    
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 3))
    plt.axis("off")
    plt.title(f"generation MNIST samples - Epoch {epoch+1}")
    
    if channels == 3:
        img_grid = grid.cpu().permute(1, 2, 0).numpy()
        img_grid = np.clip(img_grid, 0, 1)
        plt.imshow(img_grid)
    else:
        plt.imshow(grid.cpu().squeeze().numpy(), cmap='gray')
    
    sample_path = os.path.join(sample_dir, f"epoch_{epoch+1}_samples.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"样本图像已保存到: {sample_path}")
    
    for i in range(batch_size):
        single_img_path = os.path.join(sample_dir, f"epoch_{epoch+1}_sample_{i+1}.png")
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        
        if channels == 3:
            img = denorm_images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
        else:
            plt.imshow(denorm_images[i].cpu().squeeze().numpy(), cmap='gray')
        
        plt.savefig(single_img_path, dpi=150, bbox_inches='tight')
        plt.close()

def generate_conditional_samples(model, dataset, device, epoch, writer, save_dir, 
                                num_classes=10, samples_per_class=1, channels=3, image_size=32):
    print(f"生成第{epoch+1}轮的条件样本图像...")
    model.eval()
    
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            y = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)
            
            samples = model.sample_diffusion_sequence(
                batch_size=samples_per_class,
                device=device,
                y=y,
                use_ema=True
            )
            
            generated_images = samples[-1]
            all_samples.append(generated_images)
            all_labels.extend([class_idx] * samples_per_class)
    
    all_samples = torch.cat(all_samples, dim=0)
    
    denorm_images = dataset.denormalize(all_samples)
    
    grid = vutils.make_grid(denorm_images, nrow=num_classes, padding=2, normalize=False)
    
    writer.add_image(f'Conditional_Generated_Images/Epoch_{epoch+1}', grid, epoch)
    
    sample_dir = os.path.join(save_dir, "conditional_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    plt.axis("off")
    plt.title(f"条件生成的MNIST图像样本 - Epoch {epoch+1}\n数字 0-9")
    
    if channels == 3:
        img_grid = grid.cpu().permute(1, 2, 0).numpy()
        img_grid = np.clip(img_grid, 0, 1)
        plt.imshow(img_grid)
    else:
        plt.imshow(grid.cpu().squeeze().numpy(), cmap='gray')
    
    sample_path = os.path.join(sample_dir, f"epoch_{epoch+1}_conditional_samples.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"条件样本图像已保存到: {sample_path}")

def train_full_model(model, dataset, optimizer, scheduler, device, epochs, save_dir, writer):
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print('='*50)
        
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataset.get_train_loader(),
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        eval_loss = evaluate(
            model=model,
            dataloader=dataset.get_test_loader(),
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        if (epoch + 1) % 5 == 0:
            generate_and_log_samples(
                model=model,
                dataset=dataset,
                device=device,
                epoch=epoch,
                writer=writer,
                save_dir=save_dir,
                batch_size=5,
                channels=model.img_channels,
                image_size=model.img_size[0]
            )
            
            if hasattr(model.model, 'num_classes') and model.model.num_classes is not None:
                generate_conditional_samples(
                    model=model,
                    dataset=dataset,
                    device=device,
                    epoch=epoch,
                    writer=writer,
                    save_dir=save_dir,
                    num_classes=model.model.num_classes,
                    samples_per_class=2,
                    channels=model.img_channels,
                    image_size=model.img_size[0]
                )
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model_path = os.path.join(save_dir, "best_model.pt")
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=eval_loss,
                save_path=best_model_path
            )
            print(f"🎉 新的最佳模型已保存! 损失: {best_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=eval_loss,
                save_path=checkpoint_path
            )
        
        print(f"训练损失: {train_loss:.4f} | 验证损失: {eval_loss:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 32
    channels = 3
    batch_size = 16
    num_timesteps = 1000
    base_channels = 128
    num_res_blocks = 2
    dropout = 0.1
    channel_mult = (1, 2, 4, 8)
    attention_resolutions = (8, 16)
    time_emb_dim = 512
    
    use_conditional_generation = False
    num_classes = 10 if use_conditional_generation else None
    
    unet = create_unet(
        image_size=image_size,
        in_channels=channels,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        time_emb_dim=time_emb_dim,
        time_emb_scale=1.0,
        num_classes=num_classes,
        norm="gn",
        num_groups=32,
        activation=F.relu,
        initial_pad=0,
    )
    
    diffusion_model = DiffusionModel(
        model=unet,
        num_steps=num_timesteps,
        img_channels=channels,
        img_size=(image_size, image_size),
        ema_decay=0.999,
        ema_start=5000,
        ema_update_rate=1,
    )
    
    diffusion_model = diffusion_model.to(device)
    
    total_params = sum(p.numel() for p in diffusion_model.parameters())
    print(f"模型参数数量: {total_params:,}")
    print(f"设备: {device}")

    dataset = MNISTDataset(
        root_dir="./data",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        use_augmentation=True,
        convert_to_rgb=True
    )
    
    optimizer_type = "AdamW"
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    use_scheduler = True
    scheduler_type = "CosineAnnealingLR"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    epochs = 50
    save_dir = "./checkpoints"
    dataset_info = dataset.get_dataset_info()
    
    training_config = {
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "save_dir": save_dir,
        
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_scheduler": use_scheduler,
        "scheduler_type": scheduler_type if use_scheduler else None,
        
        "total_params": total_params,
        "image_size": image_size,
        "channels": channels,
        "num_timesteps": num_timesteps,
        "base_channels": base_channels,
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "dropout": dropout,
        "time_emb_dim": time_emb_dim,
        "use_conditional_generation": use_conditional_generation,
        "num_classes": num_classes,
        
        "ema_decay": diffusion_model.ema_decay,
        "ema_start": diffusion_model.ema_start,
        "ema_update_rate": diffusion_model.ema_update_rate,
        
        "train_size": dataset_info["train_size"],
        "test_size": dataset_info["test_size"],
        "use_augmentation": dataset.use_augmentation,
        "normalization": dataset_info["normalization"],
        "convert_to_rgb": dataset_info.get("convert_to_rgb", False)
    }
    
    print_training_parameters(training_config)

    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard日志保存到: {log_dir}")
    print("可以使用以下命令启动TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    config_text = "\n".join([f"{k}: {v}" for k, v in training_config.items()])
    writer.add_text("Training_Config", config_text, 0)
    
    print("\n开始训练...\n")
    
    train_full_model(
        model=diffusion_model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        writer=writer
    )
    
    print("\n训练完成! 生成最终样本...")
    generate_and_log_samples(
        model=diffusion_model,
        dataset=dataset,
        device=device,
        epoch=epochs-1,
        writer=writer,
        save_dir=save_dir,
        batch_size=10,
        channels=channels,
        image_size=image_size
    )
    
    if use_conditional_generation:
        print("生成条件样本...")
        generate_conditional_samples(
            model=diffusion_model,
            dataset=dataset,
            device=device,
            epoch=epochs-1,
            writer=writer,
            save_dir=save_dir,
            num_classes=num_classes,
            samples_per_class=3,
            channels=channels,
            image_size=image_size
        )
    
    writer.close()
    
    print(f"模型保存在: {save_dir}")
    print(f"TensorBoard日志: {log_dir}")

if __name__ == "__main__":
    main()