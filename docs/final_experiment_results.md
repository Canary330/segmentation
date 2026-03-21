# 最终实验结果

日期：2026-03-21

## 最优的申报书对齐模型

- 实验名称：`vlm_stage2_proposal_mscope_large_384`
- 文本编码器：使用 `ModelScope` 下载到本地的 `clip-vit-large-patch14`
- 初始化方式：视觉分支由 `a4c13_mobileunet_fpn_pretrained_512/best.pt` 初始化
- 第一阶段：在 `a4c13 + public_second_trimester` 上进行提示式辅助预训练
- 第二阶段：在 `a4c13` 上进行目标域精调

## 最优的 13 类测试结果

当前部署时推荐的配置如下：

- 提示词策略：对 `get_prompts_for_label(label)` 返回的全部提示词概率取平均
- 13 类重组阈值：`0.15`

测试集指标如下：

- `mean_iou_fg`：`0.4909195303916931`
- `mean_dice_fg`：`0.6474736928939819`
- `mean_iou_all`：`0.5206787586212158`
- `mean_dice_all`：`0.6691922545433044`

同一配置下的 `mAP` 指标如下：

- `mAP@50`：`0.443046398046398`
- `mAP@50:95`：`0.13897069597069597`

## 与纯视觉基线的对比

纯视觉最佳模型的测试结果如下：

- `mean_iou_fg`：`0.428771`
- `mean_dice_fg`：`0.5886`
- `mAP@50`：`0.402747`
- `mAP@50:95`：`0.129728`

当前最优的申报书对齐视觉语言模型，在上述主要前景指标上均优于纯视觉基线。

## 后端推理默认配置

推荐的后端接口环境变量如下：

```bash
SEGMENTATION_MODEL_PATH=/path/to/best.pt
SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
SEGMENTATION_IMAGE_SIZE=384
SEGMENTATION_BINARY_THRESHOLD=0.5
SEGMENTATION_MULTICLASS_THRESHOLD=0.15
```

`/predict_13class` 接口应默认对每个目标类别的全部提示词进行概率平均，再输出最终的 13 类分割结果。
