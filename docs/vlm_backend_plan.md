# 视觉语言后端完成说明

## 重新对齐申报书后的后端方案

在“拿不到新增私有数据”的前提下，后端大模型部分改为一条更稳的路线：

1. 纯视觉基线仍保留 `MobileUNet-FPN`
2. 视觉语言部分改为 `Promptable MobileUNet-FPN`
3. 训练目标改成“文本提示驱动的二值分割”
4. 公开数据与现有 XML 通过统一 prompt 数据格式联合训练
5. CLIP 文本编码器通过 LoRA 微调
6. 纯视觉最佳 `MobileUNet-FPN` 权重用于初始化融合模型的视觉分支
6. 最终评估与最终展示仍回到 A4C `13` 类
7. FastAPI 同时提供按提示词推理和 `13` 类整体验证接口

## 为什么改成提示式二值分割

你的现有 XML 是 A4C `13` 类精细标注，但公开数据标签体系并不一致。
如果坚持直接做统一 `13` 类 softmax，多源数据很难合并。

改成 prompt-conditioned binary segmentation 后：

- 公开数据里只要有任意可解释结构标签，都能作为一条“图像 + 提示词 + 二值掩码”样本
- 现有 `13` 类 XML 可以完整保留
- 系统形式也更贴近申报书里“医生输入结构名称即可自动分割”的目标
- 最终仍可通过遍历 `13` 个结构提示词，把结果重新组合为 `13` 类掩码

## 新增的主要文件

- `scripts/prepare_prompt_seg_data.py`
- `vlm_backend/label_spaces.py`
- `vlm_backend/prompt_templates.py`
- `vlm_backend/lora.py`
- `vlm_backend/model.py`
- `vlm_backend/data.py`
- `train_vlm.py`
- `export_vlm_onnx.py`
- `flutter/fastapi_backend/main.py`

## 数据来源与使用方式

### 1. 目标域数据

- 来源：`prepared/a4c13`
- 作用：最终 A4C 目标域 prompt 分割训练与验证

### 2. 公开辅助数据

- 来源：`~/Downloads/Fetal Echocardiography Second Trimester`
- 当前已识别到带标注帧 `375` 张
- 标签包括：
  `S / VD / VS / AD / AS / Ao / PA / SVC / 3VV / AoLVOTCV / AoLVOTOV`
- 其中采用如下近似规范化：
  - `S -> SPINE`
  - `VD -> RV`
  - `VS -> LV`
  - `AD -> RA`
  - `AS -> LA`
  - `Ao -> AO`

说明：
这些公开标签只作为辅助预训练信号，不作为项目最终标签体系。
项目最终标签体系仍然是你的 A4C `13` 类 XML。

## 数据准备命令

先生成目标域多类掩码：

```bash
python3 scripts/prepare_cvat_segmentation.py \
  --preset a4c13_poly \
  --dataset-root FOCUS-dataset \
  --output-root prepared/a4c13 \
  --force
```

再生成统一 prompt 数据：

```bash
python3 scripts/prepare_prompt_seg_data.py \
  --a4c-root prepared/a4c13 \
  --public-root "/Users/mico/Downloads/Fetal Echocardiography Second Trimester" \
  --output-root prepared/prompt_seg \
  --force
```

## 训练命令

阶段一，辅助预训练：

```bash
python3 train_vlm.py \
  --data-root prepared/prompt_seg \
  --experiment-dir experiments/vlm_prompt_seg_stage1 \
  --text-encoder openai/clip-vit-base-patch32 \
  --epochs 30 \
  --batch-size 8 \
  --img-size 512 \
  --lr 2e-4 \
  --train-sources a4c13,public_second_trimester \
  --eval-sources a4c13 \
  --amp
```

阶段二，目标域精调：

```bash
python3 train_vlm.py \
  --data-root prepared/prompt_seg \
  --experiment-dir experiments/vlm_prompt_seg_stage2 \
  --text-encoder openai/clip-vit-base-patch32 \
  --epochs 20 \
  --batch-size 8 \
  --img-size 512 \
  --lr 2e-4 \
  --train-sources a4c13 \
  --eval-sources a4c13 \
  --amp
```

更贴近申报书最终版的做法：

```bash
python3 train_vlm.py \
  --data-root prepared/prompt_seg \
  --experiment-dir experiments/vlm_prompt_seg_stage2 \
  --text-encoder openai/clip-vit-large-patch14 \
  --init-pure-visual-checkpoint experiments/a4c13_mobileunet_fpn/best.pt \
  --init-checkpoint experiments/vlm_prompt_seg_stage1/best.pt \
  --epochs 20 \
  --batch-size 8 \
  --img-size 512 \
  --lr 2e-4 \
  --train-sources a4c13 \
  --eval-sources a4c13 \
  --amp
```

阶段三，回到 `13` 类整体验证：

```bash
python3 evaluate_vlm_a4c13.py \
  --checkpoint experiments/vlm_prompt_seg_stage2/best.pt \
  --data-root prepared/a4c13 \
  --split validation \
  --output experiments/vlm_prompt_seg_stage2/a4c13_validation.json
```

## 推理后端

FastAPI 接口支持：

- `GET /health`
- `POST /predict`
- `POST /predict_13class`

运行前设置：

```bash
export SEGMENTATION_MODEL_PATH=/abs/path/to/best.pt
export SEGMENTATION_TEXT_ENCODER=openai/clip-vit-base-patch32
uvicorn flutter.fastapi_backend.main:app --host 0.0.0.0 --port 8000
```

`/predict` 输入：

- `image`: 上传图像
- `prompt`: 结构名称或自然语言提示

输出：

- 分割掩码的 base64 PNG
- 前景像素数
- 原图宽高

`/predict_13class` 会自动遍历 `13` 个结构提示词，返回组合后的 `13` 类掩码。

## 上云后建议的并行实验

### 轻量实验

- 文本编码器：`openai/clip-vit-base-patch32`
- 图片尺寸：`384`
- batch size：`16`
- 目标：先验证整条链路、快速找超参范围

### 主实验

- 文本编码器：`openai/clip-vit-base-patch32`
- 图片尺寸：`512`
- batch size：`8`
- LoRA rank：`8`
- 目标：作为默认提交结果

### 强化实验

- 文本编码器：更强的医学或胎儿超声 CLIP 权重
- 图片尺寸：`512` 或 `640`
- 加强增强、延长训练轮数
- 目标：追求最优 Dice / IoU

## 训练完成后应重点汇报的结果

- 目标域 A4C `13` 类验证集 Dice / IoU
- 不同提示词表述下的鲁棒性
- 公开数据预训练前后增益
- 小结构与大结构的分开分析
- 文本提示失败案例与可解释性可视化
