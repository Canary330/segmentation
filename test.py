import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# 1. 对应训练时的配置

MODEL_PATH = 'best_mobile_v2_512.pth'  # 你的模型文件
DATA_ROOT = './dataset'  # 数据集根目录
FOLDERS_TO_TEST = ['validation', 'testing']  # 要测试的文件夹

IMG_SIZE = 512
ENCODER = 'mobilenet_v2'
TARGET_TYPE = 'cardiac'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = './evaluation_results'  # 结果保存路径


# 2. 准备工作

# 创建保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🎬 加载模型: {MODEL_PATH}")
# 架构必须完全匹配训练代码
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=None,  # 推理时不需要下载预训练权重，因为会加载你微调好的
    classes=1,
    activation='sigmoid'
)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    print("✅ 模型权重加载成功！")
else:
    print(f"❌ 找不到模型文件 {MODEL_PATH}，请检查路径。")
    exit()

model.to(DEVICE)
model.eval()

# 预处理 (仅做 Resize 和 标准化，不做增强)
preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def calculate_iou(pred_mask, true_mask):
    """计算单张图片的 IoU"""
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union



# 3. 开始评估

print(f"🔥 开始评估！设备: {DEVICE}, 分辨率: {IMG_SIZE}x{IMG_SIZE}")

for folder in FOLDERS_TO_TEST:
    images_dir = os.path.join(DATA_ROOT, folder, 'images')
    masks_dir = os.path.join(DATA_ROOT, folder, 'annfiles_mask')

    # 结果保存子目录
    save_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"⚠️ 跳过 {folder} (找不到路径)")
        continue

    files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"\n📂 正在处理 {folder} 集: 共 {len(files)} 张")

    ious = []

    for img_name in tqdm(files):
        # --- 路径 ---
        img_path = os.path.join(images_dir, img_name)
        mask_name = img_name.replace(".png", f"-{TARGET_TYPE}.png")
        mask_path = os.path.join(masks_dir, mask_name)

        # --- 读取图片 ---
        original_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # --- 读取真值 Mask ---
        has_gt = False
        if os.path.exists(mask_path):
            true_mask = cv2.imread(mask_path, 0)
            true_mask = cv2.resize(true_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            true_mask = (true_mask > 127).astype(np.float32)
            has_gt = True
        else:
            true_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # --- AI 预测 ---
        # 预处理
        augmented = preprocess(image=img_rgb)
        tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_prob = model(tensor_img)
            pred_mask = pred_prob.squeeze().cpu().numpy()
            pred_binary = (pred_mask > 0.5).astype(np.float32)

        # --- 计算 IoU ---
        iou = 0.0
        if has_gt:
            iou = calculate_iou(pred_binary, true_mask)
            ious.append(iou)

        # --- 可视化绘图 ---
        # 1. 准备底图 (Resize回512以便统一显示)
        display_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

        # 2. 画 AI 预测 (红色填充)
        red_layer = np.zeros_like(display_img)
        red_layer[:, :, 2] = 255  # Red Channel
        ai_indices = pred_binary == 1
        # 叠加半透明红色
        display_img[ai_indices] = cv2.addWeighted(display_img[ai_indices], 0.6, red_layer[ai_indices], 0.4, 0)

        # 3. 画 医生真值 (绿色描边)
        if has_gt:
            true_mask_uint8 = (true_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(true_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)  # 绿色，线宽2

        # 4. 写上分数
        text = f"{img_name}"
        if has_gt:
            text += f" | IoU: {iou:.4f}"
            # 颜色逻辑：高分绿色，中等黄色，低分红色
            if iou > 0.87:
                color = (0, 255, 0)
            elif iou > 0.70:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
        else:
            text += " (No GT)"
            color = (255, 255, 255)

        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- 保存图片 ---
        cv2.imwrite(os.path.join(save_dir, img_name), display_img)

    # --- 打印统计报告 ---
    if ious:
        avg_iou = sum(ious) / len(ious)
        min_iou = min(ious)
        print("-" * 30)
        print(f"📊 {folder} 集统计结果:")
        print(f"   🌟 平均 IoU: {avg_iou:.4f}")
        print(f"   🧱 最低 IoU: {min_iou:.4f}")
        print("-" * 30)

print(f"\n✅ 全部完成！结果图片已保存在: {OUTPUT_DIR}")
