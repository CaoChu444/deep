import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from train import TransUNet  # 从paster导入TransUNet模型
import matplotlib.pyplot as plt
import random

class TestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_Segmentation.png'))
        
        # 读取并调整图像大小
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # 读取并调整掩码大小
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # 转换为张量
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        
        # 应用标准化
        if self.transform:
            image = self.transform(image)
        
        return image, mask, img_name

def calculate_metrics(pred, target):
    # 将预测结果二值化
    pred = (pred > 0.5).float()
    
    # 确保预测和目标的尺寸匹配
    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        pred = (pred > 0.5).float()
    
    # 计算交集和并集
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # 计算IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # 计算Dice系数
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    return dice.item(), iou.item()

def visualize_results(original_img, ground_truth_mask, predicted_mask, img_name, save_dir='test_results'):
    """可视化分割结果对比"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量转换为numpy数组
    original_img = original_img.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    ground_truth_mask = ground_truth_mask.cpu().numpy().squeeze()
    predicted_mask = predicted_mask.cpu().numpy().squeeze()
    
    # 反标准化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = original_img * std + mean
    original_img = np.clip(original_img, 0, 1)
    
    # 二值化预测掩码
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(original_img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 真实掩码
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('真实掩码')
    axes[1].axis('off')
    
    # 预测掩码
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title('预测掩码')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'{img_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'对比图像已保存: {save_path}')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置图像大小
    img_size = 256
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建测试数据集
    test_dataset = TestDataset(
        img_dir=r"E:/Program Files/Tencent/ISBI2016_ISIC_Part1_Test_Data",
        mask_dir=r"E:/Program Files/Tencent/ISBI2016_ISIC_Part1_Test_GroundTruth",
        transform=transform,
        img_size=img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载TransUNet模型
    model = TransUNet(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=1
    )
    model.load_state_dict(torch.load('best_model_isic.pth'))  # 使用训练时保存的模型权重
    model = model.to(device)
    model.eval()
    
    # 初始化评估指标
    total_dice = 0
    total_iou = 0
    num_samples = len(test_dataset)
    
    # 随机选择5张图像进行可视化
    indices_to_visualize = random.sample(range(num_samples), min(5, num_samples))
    visualization_count = 0
    
    # 测试循环
    with torch.no_grad():
        for batch_idx, (images, masks, img_names) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算评估指标
            dice, iou = calculate_metrics(outputs, masks)
            total_dice += dice
            total_iou += iou
            
            # 可视化选定的图像
            if batch_idx in indices_to_visualize and visualization_count < 5:
                img_name = img_names[0].replace('.jpg', '')
                visualize_results(
                    images[0], 
                    masks[0], 
                    outputs[0], 
                    img_name
                )
                visualization_count += 1
                print(f'可视化图像 {visualization_count}/5: {img_name}')
    
    # 计算平均指标
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    
    print(f'\n测试集评估结果:')
    print(f'平均Dice系数: {avg_dice:.4f}')
    print(f'平均IoU: {avg_iou:.4f}')
    print(f'对比图像已保存到 test_results 文件夹中')

if __name__ == '__main__':
    main()
