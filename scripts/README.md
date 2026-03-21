# scripts 目录说明

这个目录存放的是数据准备脚本，作用是把原始数据整理成训练代码能直接读取的格式。

如果没有这个目录，后面的训练基本跑不起来，因为：

- XML 标注不能直接喂给模型
- 公开数据和自有数据格式也不一致

所以这个目录本质上是“数据入口层”。

## 当前目录里的文件

### `prepare_cvat_segmentation.py`

这个脚本负责把你当前的 XML 标注数据转换成标准语义分割数据集。

它会把原始 XML 和图像整理成：

- `prepared/a4c13/training/images`
- `prepared/a4c13/training/masks`
- `prepared/a4c13/validation/images`
- `prepared/a4c13/validation/masks`
- `prepared/a4c13/testing/images`
- `prepared/a4c13/testing/masks`

它的作用是：

- 把原始多边形标注转成像素级 mask
- 固定 `13` 类标签顺序
- 给纯视觉训练和最终 `13` 类评估提供标准输入

### `prepare_prompt_seg_data.py`

这个脚本负责把目标域数据和公开数据统一成“图像 + 提示词 + 二值 mask”的提示式训练格式。

它会生成：

- `prepared/prompt_seg/training.jsonl`
- `prepared/prompt_seg/validation.jsonl`
- `prepared/prompt_seg/testing.jsonl`

它的作用是：

- 把 `13` 类目标域数据转换成提示式样本
- 把公开辅助数据映射到统一标签空间
- 为 `train_vlm.py` 提供可直接训练的数据

## 这个目录在整个项目中的位置

整体流程中，这个目录是训练前的第一步：

1. 先运行 `prepare_cvat_segmentation.py`
2. 再运行 `prepare_prompt_seg_data.py`
3. 之后才进入 `train.py` 和 `train_vlm.py`

如果以后更换服务器、重做实验或者新增数据，这个目录通常都是最先需要运行的部分。

## 对应的主要入口

- [prepare_cvat_segmentation.py](/Users/mico/Documents/za/git/segmentation/scripts/prepare_cvat_segmentation.py)
- [prepare_prompt_seg_data.py](/Users/mico/Documents/za/git/segmentation/scripts/prepare_prompt_seg_data.py)
- 总复现说明：[README.md](/Users/mico/Documents/za/git/segmentation/README.md)
