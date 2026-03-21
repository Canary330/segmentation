# 顶层 Python 文件说明

这个文件专门解释仓库根目录下那些“不在文件夹里”的 Python 文件分别是干什么的。

当前根目录下的 Python 文件有：

- `train.py`
- `train_vlm.py`
- `evaluate_vlm_a4c13.py`
- `export_vlm_onnx.py`
- `test.py`

## `train.py`

这是纯视觉基线训练入口。

它的作用是：

- 读取 `prepared/a4c13`
- 训练 `MobileUNet-FPN`
- 输出纯视觉最佳权重和训练历史

什么时候用它：

- 想复现纯视觉基线
- 想重新得到视觉初始化权重
- 想做纯视觉对照实验

对应模块：

- [pure_visual/mobileunet_fpn.py](/Users/mico/Documents/za/git/segmentation/pure_visual/mobileunet_fpn.py)

## `train_vlm.py`

这是提示式视觉语言模型训练入口，也是当前项目后端大模型部分最重要的训练脚本。

它的作用是：

- 读取 `prepared/prompt_seg`
- 训练 `Promptable MobileUNet-FPN`
- 支持 `CLIP` 文本编码器
- 支持 `LoRA` 微调
- 支持加载纯视觉权重初始化
- 支持先辅助预训练、再目标域精调

什么时候用它：

- 想复现当前最优申报书对齐结果
- 想继续做多模态实验

对应模块：

- [vlm_backend/model.py](/Users/mico/Documents/za/git/segmentation/vlm_backend/model.py)
- [vlm_backend/data.py](/Users/mico/Documents/za/git/segmentation/vlm_backend/data.py)

## `evaluate_vlm_a4c13.py`

这是提示式模型回到 `13` 类整体评估时的脚本。

它的作用是：

- 对每个结构跑 prompt 分割
- 把结果重新组合成 `13` 类掩码
- 计算 `IoU`、`Dice` 等指标

什么时候用它：

- 想验证某个 `best.pt` 在 `13` 类测试集上的表现
- 想生成最终汇报用指标

## `export_vlm_onnx.py`

这是把提示式模型导出成 `ONNX` 的脚本。

它的作用是：

- 读取训练好的提示式模型权重
- 导出部署格式模型

什么时候可能用它：

- 后续想做跨平台部署
- 想把模型接到别的推理环境

当前项目里它不是主线必需文件，但建议保留，因为后续部署可能会用到。

## `test.py`

这个文件不是当前主线实验的一部分，更像是早期遗留的独立测试脚本。

从内容看，它是：

- 基于 `segmentation_models_pytorch`
- 用旧数据目录结构做评估和可视化
- 针对单一类别或旧任务流程

它和当前主线的关系是：

- 不是当前最优结果的生成脚本
- 不参与当前 `13` 类提示式模型训练
- 不参与当前前后端联调

所以现在的建议是：

- 可以保留，作为历史参考
- 但不要把它当成当前主线复现入口

## 推荐优先级

如果你现在是为了继续项目，建议按下面顺序关注这些文件：

1. `train_vlm.py`
2. `train.py`
3. `evaluate_vlm_a4c13.py`
4. `export_vlm_onnx.py`
5. `test.py`

也就是说：

- 前三个是主线
- 第四个是可选部署工具
- 第五个是历史遗留参考
