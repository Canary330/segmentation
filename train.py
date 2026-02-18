import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. 压榨性能配置

TRAIN_DIR = './dataset/training'
VAL_DIR = './dataset/validation'
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
ACTIVATION = 'sigmoid'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ⚡️ 核心改动 1: 分辨率提升到 512
IMG_SIZE = 512
# ⚡️ 核心改动 2: 显存不够就改小这个 (比如 4)
BATCH_SIZE = 4
LR = 0.0003

# ⚡️ 修改点 1: 轮数改为 200
EPOCHS = 200
TARGET_TYPE = 'cardiac'


# 2. 增强策略 (保持强力)

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.5),
    A.GaussNoise(p=0.2),
    A.OpticalDistortion(distort_limit=0.05, p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
    ], p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ============================
# 3. 数据集 (Dataset)
# ============================
class UltimateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annfiles_mask')
        self.ids = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.replace(".png", f"-{TARGET_TYPE}.png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not os.path.exists(mask_path):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, 0)

        mask = mask.astype('float32')
        mask[mask < 127] = 0.0
        mask[mask >= 127] = 1.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)
        return image, mask

    def __len__(self):
        return len(self.ids)


def calculate_iou(pred_mask, true_mask):
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    if union == 0: return 1.0
    return intersection / union



# 4. 主程序

if __name__ == '__main__':
    train_ds = UltimateDataset(TRAIN_DIR, transform=train_transform)
    val_ds = UltimateDataset(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"🔥 MobileNetV2 极限挑战开始！分辨率: {IMG_SIZE}x{IMG_SIZE}, 目标: Min IoU > 0.87")

    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION
    )
    model.to(DEVICE)

    loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()

        # ⚡️ 修改点 2: 改为收集所有单张图片的 IoU，以便计算 Min IoU
        val_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)

                # 必须计算 Batch 中每一张图的 IoU，而不是整个 Batch 的 IoU
                for i in range(images.size(0)):
                    single_pred = outputs[i:i + 1]
                    single_mask = masks[i:i + 1]
                    single_iou = calculate_iou(single_pred, single_mask).item()
                    val_ious.append(single_iou)

        scheduler.step()

        # 计算统计指标
        avg_val_iou = sum(val_ious) / len(val_ious)
        min_val_iou = min(val_ious)  # 找出这一轮的“最短板”

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} | Val IoU: {avg_val_iou:.4f} | Min IoU: {min_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), 'best_mobile_v2_512.pth')
            print(f"    🏆 新纪录！Avg IoU: {best_iou:.4f} (Min: {min_val_iou:.4f})")

    print(f"✅ 挑战结束。最高 Avg IoU: {best_iou:.4f}")