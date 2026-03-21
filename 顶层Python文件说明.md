# 关键 Python 入口文件说明

仓库重构后，主线 Python 入口基本都已经集中到 `backend/`。

当前最重要的入口文件有：

- `backend/train.py`
- `backend/train_vlm.py`
- `backend/evaluate_vlm_a4c13.py`
- `backend/export_vlm_onnx.py`
- `test.py`

## `backend/train.py`

这是纯视觉 `MobileUNet-FPN` 的训练入口。

作用：

- 读取 `prepared/a4c13`
- 训练纯视觉 `13` 类分割模型
- 输出纯视觉最佳权重

适用场景：

- 复现纯视觉基线
- 重新生成视觉初始化权重
- 做对照实验

## `backend/train_vlm.py`

这是提示式视觉语言模型训练入口，也是当前后端大模型主线训练脚本。

作用：

- 读取 `prepared/prompt_seg`
- 训练 `Promptable MobileUNet-FPN`
- 接入文本编码器
- 做 `LoRA` 微调
- 支持纯视觉权重初始化
- 支持辅助预训练和目标域精调

适用场景：

- 复现当前最优结果
- 继续做申报书路线实验

## `backend/evaluate_vlm_a4c13.py`

这是把提示式模型重新组合回 A4C `13` 类并计算指标的评估入口。

作用：

- 遍历 `13` 个结构提示词
- 生成 `13` 类掩码
- 计算 `IoU`、`Dice`

适用场景：

- 产出最终汇报用结果
- 验证某个 `best.pt` 是否达到预期

## `backend/export_vlm_onnx.py`

这是提示式模型导出 `ONNX` 的入口。

作用：

- 读取训练好的提示式模型
- 导出部署格式模型

适用场景：

- 想做额外部署尝试
- 想接别的推理系统

这个文件不是当前主线必需，但建议保留。

## `test.py`

这是旧流程遗留的独立测试脚本，不属于当前主线。

特点：

- 使用旧数据结构
- 使用旧的单任务评估方式
- 不参与当前前后端联调

建议：

- 保留作历史参考
- 不要把它当成当前项目的正式入口

## 推荐关注顺序

1. `backend/train_vlm.py`
2. `backend/train.py`
3. `backend/evaluate_vlm_a4c13.py`
4. `backend/export_vlm_onnx.py`
5. `test.py`
