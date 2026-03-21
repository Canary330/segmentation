# 胎儿超声 13 类语义分割项目复现说明

## 1. 项目目标

本项目最终任务是：

- 输入胎儿四腔心超声图像
- 支持医生输入文字提示词，完成指定结构的语义分割
- 同时支持回到申报书要求的 `13` 类整体分割结果

最终标签体系固定为：

- `DAO`
- `LA`
- `RA`
- `LV`
- `RV`
- `VS`
- `IS`
- `SP`
- `RB`
- `LVW`
- `RVW`
- `LL`
- `RL`

当前最优路线不是“只做纯视觉”，而是：

1. 先训练纯视觉 `MobileUNet-FPN` 基线
2. 再用纯视觉最佳权重初始化提示式视觉语言模型
3. 用公开数据做辅助预训练
4. 回到 A4C `13` 类目标域精调
5. 最终评估与部署仍然回到 `13` 类

## 2. 当前最优结果概览

当前最优实验名称：

- `vlm_stage2_proposal_mscope_large_384`

当前最优模型配置：

- 视觉骨架：`Promptable MobileUNet-FPN`
- 文本编码器：`clip-vit-large-patch14`
- 文本编码器来源：`ModelScope` 本地缓存
- 视觉初始化：`a4c13_mobileunet_fpn_pretrained_512/best.pt`
- 推理方式：每个类别使用多提示词概率平均
- 最终 `13` 类阈值：`0.15`

当前最优 `13` 类测试集平均指标：

| 指标 | 数值 |
| --- | ---: |
| `mean_iou_fg` | `0.4909195303916931` |
| `mean_dice_fg` | `0.6474736928939819` |
| `mean_iou_all` | `0.5206787586212158` |
| `mean_dice_all` | `0.6691922545433044` |
| `mAP@50` | `0.443046398046398` |
| `mAP@50:95` | `0.13897069597069597` |

纯视觉最佳基线结果：

| 指标 | 数值 |
| --- | ---: |
| `mean_iou_fg` | `0.428771` |
| `mean_dice_fg` | `0.5886` |
| `mAP@50` | `0.402747` |
| `mAP@50:95` | `0.129728` |

结论：

- 当前最优的申报书对齐模型已经超过纯视觉基线
- 当前部署应优先使用 `vlm_stage2_proposal_mscope_large_384`

## 3. 当前最优结果的详细指标

下表对应文件：

- [`server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json)

| 类别 | IoU | Dice | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: | ---: |
| `DAO` | `0.520332` | `0.684498` | `0.414286` | `0.122857` |
| `LA` | `0.623265` | `0.767916` | `0.722222` | `0.238056` |
| `RA` | `0.668253` | `0.801141` | `0.788889` | `0.342000` |
| `LV` | `0.544586` | `0.705154` | `0.571429` | `0.171429` |
| `RV` | `0.460863` | `0.630946` | `0.483333` | `0.125833` |
| `VS` | `0.515065` | `0.679924` | `0.700000` | `0.085000` |
| `IS` | `0.434029` | `0.605328` | `0.166667` | `0.033333` |
| `SP` | `0.432327` | `0.603671` | `0.187500` | `0.050000` |
| `RB` | `0.465609` | `0.635380` | `0.271111` | `0.030444` |
| `LVW` | `0.240793` | `0.388127` | `0.000000` | `0.000000` |
| `RVW` | `0.239822` | `0.386865` | `0.016667` | `0.001667` |
| `LL` | `0.598551` | `0.748867` | `0.637500` | `0.291250` |
| `RL` | `0.638460` | `0.779342` | `0.800000` | `0.314750` |
| `13 类平均` | `0.490920` | `0.647474` | `0.443046` | `0.138971` |

说明：

- `LVW` 和 `RVW` 仍然是最难类别
- `RA`、`RL`、`LA`、`LL` 是当前表现最稳定的结构
- 这组结果已经是当前项目推荐汇报和部署使用的主结果

## 4. 已经拉回本地的重要产物

这些文件已经回到本地，后续换服务器时可以直接使用：

- 最优权重：
  [`server_artifacts/20260321_proposal_best/files/best.pt`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/best.pt)
- 训练历史：
  [`server_artifacts/20260321_proposal_best/files/history.json`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/history.json)
- `13` 类测试指标：
  [`server_artifacts/20260321_proposal_best/files/a4c13_eval_testing_tuned.json`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/a4c13_eval_testing_tuned.json)
- 含 `mAP` 的最终指标：
  [`server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json)
- 整包归档：
  [`server_artifacts/20260321_proposal_best/server_artifacts_20260321_proposal_best.tgz`](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/server_artifacts_20260321_proposal_best.tgz)

如果只是更换服务器做部署，不想重新训练，可以直接把以上产物传到新服务器使用。

## 5. 从零开始拿到服务器，如何完整复现当前结果

这一节按“新的 AutoDL / Linux GPU 服务器”来写。

### 5.1 服务器建议配置

最低建议：

- Ubuntu 20.04 或 22.04
- Python `3.10`
- CUDA 可用
- 显存建议 `>= 24 GB`
- 磁盘建议 `>= 80 GB`

### 5.2 拉取代码

```bash
cd /root/autodl-tmp/project
git clone <你的仓库地址> segmentation
cd segmentation
```

如果不是用 `git clone`，也可以直接把当前项目目录整体上传到服务器。

### 5.3 安装训练环境

建议使用独立环境。

```bash
cd /root/autodl-tmp/project/segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
pip install modelscope
```

如果服务器已经有合适的 `conda` 环境，也可以直接安装以上依赖。

### 5.4 准备数据

需要两部分数据：

1. 你的目标域 `13` 类 XML 标注数据
2. 公开辅助数据 `Fetal Echocardiography Second Trimester`

推荐在服务器上整理成如下结构：

```text
/root/autodl-tmp/project/segmentation/
├── FOCUS-dataset/
│   ├── training2/
│   ├── validation2/
│   ├── testing2/
│   ├── training2.xml
│   ├── validation2.xml
│   └── testing2.xml
├── public-datasets/
│   └── Fetal Echocardiography Second Trimester/
└── ...
```

其中：

- `FOCUS-dataset/` 是你当前 XML 对应的数据目录
- `public-datasets/Fetal Echocardiography Second Trimester/` 是公开辅助数据

### 5.5 只用国内镜像下载文本编码器

不要直接走外网 Hugging Face。

当前推荐用 `ModelScope` 把文本编码器下载到本地目录：

```bash
python3 - <<'PY'
from modelscope import snapshot_download
path = snapshot_download(
    'AI-ModelScope/clip-vit-large-patch14',
    cache_dir='/root/autodl-tmp/modelscope_cache'
)
print(path)
PY
```

下载完成后，通常会得到类似目录：

```text
/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
```

后续训练和推理都直接使用这个本地路径。

### 5.6 生成 `13` 类目标域数据

```bash
python3 scripts/prepare_cvat_segmentation.py \
  --preset a4c13_poly \
  --dataset-root FOCUS-dataset \
  --output-root prepared/a4c13 \
  --force
```

生成完成后，重点检查：

- `prepared/a4c13/training`
- `prepared/a4c13/validation`
- `prepared/a4c13/testing`
- `prepared/a4c13/summary.json`

### 5.7 生成统一的提示式训练数据

```bash
python3 scripts/prepare_prompt_seg_data.py \
  --a4c-root prepared/a4c13 \
  --public-root /root/autodl-tmp/project/segmentation/public-datasets/"Fetal Echocardiography Second Trimester" \
  --output-root prepared/prompt_seg \
  --force
```

生成完成后，重点检查：

- `prepared/prompt_seg/training.jsonl`
- `prepared/prompt_seg/validation.jsonl`
- `prepared/prompt_seg/testing.jsonl`
- `prepared/prompt_seg/summary.json`

### 5.8 先训练纯视觉 `MobileUNet-FPN` 基线

```bash
python3 train.py \
  --data-root prepared/a4c13 \
  --experiment-dir experiments/a4c13_mobileunet_fpn_pretrained_512 \
  --epochs 120 \
  --batch-size 8 \
  --img-size 512 \
  --lr 3e-4 \
  --pretrained-backbone \
  --amp
```

这一阶段的作用：

- 得到纯视觉对照组
- 得到后续视觉语言模型初始化所需的视觉权重

最重要的输出文件是：

- `experiments/a4c13_mobileunet_fpn_pretrained_512/best.pt`

### 5.9 训练提示式视觉语言模型第一阶段

第一阶段使用目标域和公开数据联合训练。

```bash
python3 train_vlm.py \
  --data-root prepared/prompt_seg \
  --experiment-dir experiments/vlm_stage1_proposal_mscope_large_384 \
  --text-encoder /root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 \
  --epochs 30 \
  --batch-size 8 \
  --img-size 384 \
  --lr 2e-4 \
  --train-sources a4c13,public_second_trimester \
  --eval-sources a4c13 \
  --init-pure-visual-checkpoint experiments/a4c13_mobileunet_fpn_pretrained_512/best.pt \
  --amp
```

这一阶段的目标：

- 让模型获得公开数据的辅助结构先验
- 同时把纯视觉能力带入多模态模型

### 5.10 训练提示式视觉语言模型第二阶段

第二阶段只保留目标域 `a4c13`，做最终精调。

```bash
python3 train_vlm.py \
  --data-root prepared/prompt_seg \
  --experiment-dir experiments/vlm_stage2_proposal_mscope_large_384 \
  --text-encoder /root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 \
  --epochs 20 \
  --batch-size 8 \
  --img-size 384 \
  --lr 2e-4 \
  --train-sources a4c13 \
  --eval-sources a4c13 \
  --init-pure-visual-checkpoint experiments/a4c13_mobileunet_fpn_pretrained_512/best.pt \
  --init-checkpoint experiments/vlm_stage1_proposal_mscope_large_384/best.pt \
  --amp
```

这一阶段的核心输出是：

- `experiments/vlm_stage2_proposal_mscope_large_384/best.pt`
- `experiments/vlm_stage2_proposal_mscope_large_384/history.json`

### 5.11 评估 `13` 类结果

先跑验证集与测试集的 `13` 类评估：

```bash
python3 evaluate_vlm_a4c13.py \
  --checkpoint experiments/vlm_stage2_proposal_mscope_large_384/best.pt \
  --data-root prepared/a4c13 \
  --split validation \
  --text-encoder /root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 \
  --img-size 384 \
  --output experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_validation.json
```

```bash
python3 evaluate_vlm_a4c13.py \
  --checkpoint experiments/vlm_stage2_proposal_mscope_large_384/best.pt \
  --data-root prepared/a4c13 \
  --split testing \
  --text-encoder /root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 \
  --img-size 384 \
  --output experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_testing.json
```

如果你的目标是尽快对齐当前项目结果，至少确认以下文件已经生成：

- `experiments/vlm_stage2_proposal_mscope_large_384/best.pt`
- `experiments/vlm_stage2_proposal_mscope_large_384/history.json`
- `experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_validation.json`
- `experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_testing.json`

### 5.12 部署云端后端并做前端测试

云端启动后端：

```bash
export SEGMENTATION_MODEL_PATH=/root/autodl-tmp/project/segmentation/experiments/vlm_stage2_proposal_mscope_large_384/best.pt
export SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
export SEGMENTATION_IMAGE_SIZE=384
export SEGMENTATION_BINARY_THRESHOLD=0.5
export SEGMENTATION_MULTICLASS_THRESHOLD=0.15
uvicorn flutter.fastapi_backend.main:app --host 0.0.0.0 --port 8000
```

本地建立隧道：

```bash
ssh -p <服务器端口> -L 18000:127.0.0.1:8000 root@<服务器地址>
```

本地启动 Flutter：

```bash
cd flutter/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

## 6. 最快复现路径

如果你不是为了重新训练，只是为了快速部署、测试和演示，建议直接走下面这条最快路径：

1. 把本地最优权重和结果包传到新服务器
2. 仅在新服务器准备 Python 运行环境
3. 下载 `ModelScope` 文本编码器到本地缓存
4. 直接启动 `FastAPI` 后端
5. 本地通过 `Flutter` 前端连接并测试

这条路径最快，因为不需要重复训练。

需要带走的文件是：

- `server_artifacts/20260321_proposal_best/files/best.pt`
- `server_artifacts/20260321_proposal_best/files/history.json`
- `server_artifacts/20260321_proposal_best/files/test_metrics_with_map_ensemble_tuned.json`

## 7. 拿到结果后，如何完成整个实验

拿到 `best.pt` 和最终评估文件后，不要只停在“模型训练结束”，还需要把整个实验闭环做完。

建议按下面顺序收尾：

### 7.1 固化最终结果

把以下文件统一归档到一个目录：

- `best.pt`
- `history.json`
- `a4c13_eval_validation.json`
- `a4c13_eval_testing.json`
- `test_metrics_with_map_ensemble_tuned.json`
- 可视化图片

### 7.2 生成汇报图

至少生成以下可视化：

1. 单提示词分割效果图
2. `13` 类整体分割效果图
3. 原图与输出叠加图
4. 测试集代表性成功案例
5. 测试集失败案例

### 7.3 做对照实验表

最终报告中至少要有三组对比：

1. 纯视觉 `MobileUNet-FPN`
2. 视觉语言模型初版
3. 当前最优申报书对齐版本

如果时间允许，再补两类分析：

1. 有无公开数据辅助预训练
2. 有无纯视觉权重初始化

### 7.4 写清楚最终结论

最终结论建议这样表述：

- 项目最终仍然围绕 A4C `13` 类结构分割展开，没有偏离申报书主题
- 公开数据只作为辅助预训练信号，不改变最终标签体系
- 当前最优的提示式视觉语言模型已经超过纯视觉基线
- 小结构仍然是主要难点，后续可从数据扩充和更强特征对齐继续优化

### 7.5 做可演示系统

当前仓库已经具备：

- `FastAPI` 推理后端
- `Flutter` 可视化测试前端

建议在答辩或验收前至少验证以下功能：

1. 检查后端
2. 单提示词分割
3. `13` 类整体分割
4. 叠加预览
5. 导出 `mask`
6. 导出预览图

## 8. 复现完成的判定标准

满足以下条件，就算成功复现当前项目主结果：

1. `prepared/a4c13` 和 `prepared/prompt_seg` 均成功生成
2. 纯视觉模型 `best.pt` 成功生成
3. `vlm_stage2_proposal_mscope_large_384/best.pt` 成功生成
4. `13` 类评估文件成功生成
5. 后端接口 `/health`、`/predict`、`/predict_13class` 都能正常工作
6. Flutter 前端能够正常连通后端并展示结果

## 9. 相关文件

建议优先阅读以下文件：

- [docs/final_experiment_results.md](/Users/mico/Documents/za/git/segmentation/docs/final_experiment_results.md)
- [docs/vlm_backend_plan.md](/Users/mico/Documents/za/git/segmentation/docs/vlm_backend_plan.md)
- [flutter/README.md](/Users/mico/Documents/za/git/segmentation/flutter/README.md)
- [scripts/prepare_cvat_segmentation.py](/Users/mico/Documents/za/git/segmentation/scripts/prepare_cvat_segmentation.py)
- [scripts/prepare_prompt_seg_data.py](/Users/mico/Documents/za/git/segmentation/scripts/prepare_prompt_seg_data.py)
- [train.py](/Users/mico/Documents/za/git/segmentation/train.py)
- [train_vlm.py](/Users/mico/Documents/za/git/segmentation/train_vlm.py)
- [evaluate_vlm_a4c13.py](/Users/mico/Documents/za/git/segmentation/evaluate_vlm_a4c13.py)
- [flutter/fastapi_backend/main.py](/Users/mico/Documents/za/git/segmentation/flutter/fastapi_backend/main.py)

---

如果是新同学接手这个项目，最推荐的顺序是：

1. 先看本文件
2. 直接把后端跑起来验证现有最优模型
3. 再复现完整训练
4. 最后做汇报图和实验总结
