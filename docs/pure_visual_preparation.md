# 纯视觉部分准备说明

## 当前训练口径

- 申报书纯视觉基线使用 `MobileUNet-FPN`
- 训练目标采用 A4C 的 `13` 类解剖结构：
  `DAO / LA / RA / LV / RV / VS / IS / SP / RB / LVW / RVW / LL / RL`
- 当前最适合直接训练这 13 类任务的是：
  `标注/training2.xml`、`标注/validation2.xml`、`标注/testing2.xml`

## 已核对的数据情况

- `training2.xml`：25 张图，含 368 个 polygon 标注
- `validation2.xml`：15 张图，含 222 个 polygon 标注
- `testing2.xml`：10 张图，含 145 个 polygon 标注
- 旧版 `training.xml/testing.xml` 主要是 8 类 mask 标注，和申报书 13 类口径不一致，因此不作为当前主训练集

## 生成训练数据

先把 CVAT XML 转成语义分割掩码：

```bash
python3 scripts/prepare_cvat_segmentation.py \
  --preset a4c13_poly \
  --dataset-root FOCUS-dataset \
  --output-root prepared/a4c13 \
  --force
```

生成后目录结构如下：

```text
prepared/a4c13/
  summary.json
  training/
    images/
    masks/
  validation/
    images/
    masks/
  testing/
    images/
    masks/
```

## 训练命令

```bash
python3 train.py \
  --data-root prepared/a4c13 \
  --experiment-dir experiments/a4c13_mobileunet_fpn \
  --epochs 120 \
  --batch-size 8 \
  --img-size 512 \
  --lr 3e-4 \
  --pretrained-backbone \
  --amp
```

## 产物说明

- `experiments/a4c13_mobileunet_fpn/best.pt`：最佳验证集权重
- `experiments/a4c13_mobileunet_fpn/last.pt`：最后一轮权重
- `experiments/a4c13_mobileunet_fpn/history.json`：逐轮训练记录
- `experiments/a4c13_mobileunet_fpn/test_metrics.json`：最佳权重在测试集上的结果

## 上云前检查

- 安装 `requirements-train.txt`
- 确认 GPU 可用
- 把 `FOCUS-dataset/` 和 `标注/` 同步到云端
- 先运行一次数据转换脚本，再启动训练
