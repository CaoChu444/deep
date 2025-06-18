import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt

# ----------------- 数据增强 -----------------
class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[1:]
        new_h, new_w = self.output_size
        if h < new_h or w < new_w:
            scale = max(new_h / h, new_w / w)
            new_h_orig = int(h * scale)
            new_w_orig = int(w * scale)
            image = cv2.resize(image.transpose(1,2,0), (new_w_orig, new_h_orig), interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
            mask = cv2.resize(mask[0], (new_w_orig, new_h_orig), interpolation=cv2.INTER_NEAREST)[None, ...]
            h, w = new_h_orig, new_w_orig
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[:, top:top + new_h, left:left + new_w]
        mask = mask[:, top:top + new_h, left:left + new_w]
        return {'image': image, 'mask': mask}

class RandomAffine:
    def __init__(self, degrees=10, translate=0.1, scale=(0.9, 1.1), shear=10):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        c, h, w = image.shape
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        M[0,2] += tx
        M[1,2] += ty
        image = cv2.warpAffine(image.transpose(1,2,0), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).transpose(2,0,1)
        mask = cv2.warpAffine(mask[0], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)[None, ...]
        return {'image': image, 'mask': mask}

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.08):
        self.std = std
        self.mean = mean
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        noise = np.random.normal(self.mean, self.std, image.shape)
        image = image + noise
        image = np.clip(image, 0, 1)
        return {'image': image, 'mask': mask}

class ToTensor:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': torch.from_numpy(image).float(), 'mask': torch.from_numpy(mask).float()}

# ----------------- ISIC数据集 -----------------
class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.img_list, self.mask_list = self._pair_files()
    def _pair_files(self):
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('_Segmentation.png')])
        img_paths, mask_paths = [], []
        for img in img_files:
            img_id = os.path.splitext(img)[0]
            mask_name = img_id + '_Segmentation.png'
            if mask_name in mask_files:
                img_paths.append(os.path.join(self.img_dir, img))
                mask_paths.append(os.path.join(self.mask_dir, mask_name))
        return img_paths, mask_paths
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_list[idx]).convert('RGB')) / 255.0
        mask = np.array(Image.open(self.mask_list[idx]).convert('L'))
        mask = (mask > 127).astype(np.float32)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = img.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=0)
        sample = {'image': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

# ----------------- TransUNet（简化版）-----------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, mlp_dim=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        return self.encoder(x)

class TransUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256, num_heads=4, num_layers=2, num_classes=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.transformer = SimpleTransformerEncoder(embed_dim, num_heads, num_layers, mlp_dim=embed_dim*4)
        self.decoder_conv1 = nn.ConvTranspose2d(embed_dim, 128, 2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
    def forward(self, x):
        B = x.shape[0]
        x_patch = self.patch_embed(x)
        x_patch = x_patch + self.pos_embed
        x_trans = self.transformer(x_patch)
        h = w = self.img_size // self.patch_size
        x_feat = x_trans.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        x = nn.functional.relu(self.decoder_conv1(x_feat))
        x = nn.functional.relu(self.decoder_conv2(x))
        x = nn.functional.relu(self.decoder_conv3(x))
        x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return torch.sigmoid(x)

# ----------------- 训练主流程 -----------------
def main():
    img_dir = r"E:/Program Files/Tencent/ISBI2016_ISIC_Part1_Training_Data"
    mask_dir = r"E:/Program Files/Tencent/ISBI2016_ISIC_Part1_Training_GroundTruth"
    img_size = 256
    batch_size = 4
    num_epochs = 40
    lr = 2e-4
    # 划分训练/验证集
    dataset = ISICDataset(img_dir, mask_dir, img_size=img_size)
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = int(n * 0.2)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    # 数据增强
    train_transform = transforms.Compose([
        RandomCrop(img_size),
        RandomAffine(degrees=10, translate=0.1, scale=(0.9, 1.1), shear=10),
        AddGaussianNoise(0., 0.08),
        ToTensor()
    ])
    val_transform = transforms.Compose([
        RandomCrop(img_size),
        ToTensor()
    ])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            bce_loss = nn.BCELoss()(outputs, masks)
            dice_loss = 1 - (2. * (outputs * masks).sum() + 1e-8) / (outputs.sum() + masks.sum() + 1e-8)
            loss = bce_loss + dice_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                outputs = model(images)
                bce_loss = nn.BCELoss()(outputs, masks)
                dice_loss = 1 - (2. * (outputs * masks).sum() + 1e-8) / (outputs.sum() + masks.sum() + 1e-8)
                loss = bce_loss + dice_loss
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_isic.pth')
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    # 损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('isic_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
