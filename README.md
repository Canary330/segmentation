# 胎儿超声 13 类语义分割项目

这个仓库现在已经重构成真正的前后端分离结构：

- `backend/`：训练、评估、推理后端
- `frontend/`：Flutter 测试前端
- `docs/`：部署、复现、实验说明
- `server_artifacts/`：已经从云端拉回本地的实验结果与权重

当前推荐直接使用的最优权重在：

- [best.pt](/Users/mico/Documents/za/git/segmentation/server_artifacts/20260321_proposal_best/files/best.pt)

当前推荐直接阅读的文档是：

- [docs/前后端部署与服务器启动说明.md](/Users/mico/Documents/za/git/segmentation/docs/前后端部署与服务器启动说明.md)
- [docs/final_experiment_results.md](/Users/mico/Documents/za/git/segmentation/docs/final_experiment_results.md)

## 仓库结构

```text
segmentation/
├── backend/
│   ├── api/
│   ├── pure_visual/
│   ├── scripts/
│   ├── vlm_backend/
│   ├── train.py
│   ├── train_vlm.py
│   ├── evaluate_vlm_a4c13.py
│   ├── export_vlm_onnx.py
│   └── requirements.txt
├── frontend/
│   ├── flutter_frontend/
│   └── README.md
├── docs/
├── prepared/
├── server_artifacts/
└── FOCUS-dataset/
```

## 当前最优结果

当前最优实验：

- `vlm_stage2_proposal_mscope_large_384`

当前最优 `13` 类测试结果：

| 指标 | 数值 |
| --- | ---: |
| `mean_iou_fg` | `0.4909195303916931` |
| `mean_dice_fg` | `0.6474736928939819` |
| `mean_iou_all` | `0.5206787586212158` |
| `mean_dice_all` | `0.6691922545433044` |
| `mAP@50` | `0.443046398046398` |
| `mAP@50:95` | `0.13897069597069597` |

详细分类别指标见：

- [docs/final_experiment_results.md](/Users/mico/Documents/za/git/segmentation/docs/final_experiment_results.md)

## 最短启动方式

如果你只是想把当前最优结果直接跑起来，不重新训练，最短步骤如下。

### 1. 服务器启动后端

```bash
cd /root/autodl-tmp/project/segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install modelscope
```

准备文本编码器：

```bash
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'AI-ModelScope/clip-vit-large-patch14',
    cache_dir='/root/autodl-tmp/modelscope_cache'
)
PY
```

启动后端：

```bash
export SEGMENTATION_MODEL_PATH=/root/autodl-tmp/project/segmentation/server_artifacts/20260321_proposal_best/files/best.pt
export SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
export SEGMENTATION_IMAGE_SIZE=384
export SEGMENTATION_BINARY_THRESHOLD=0.5
export SEGMENTATION_MULTICLASS_THRESHOLD=0.15
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

### 2. 本地开端口转发

```bash
ssh -p <服务器端口> -L 18000:127.0.0.1:8000 root@<服务器地址>
```

本地后端地址固定填：

```text
http://127.0.0.1:18000
```

### 3. 本地启动前端

```bash
cd frontend/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

## 重新训练的主要入口

- 纯视觉基线训练：[`backend/train.py`](/Users/mico/Documents/za/git/segmentation/backend/train.py)
- 提示式视觉语言训练：[`backend/train_vlm.py`](/Users/mico/Documents/za/git/segmentation/backend/train_vlm.py)
- `13` 类评估：[`backend/evaluate_vlm_a4c13.py`](/Users/mico/Documents/za/git/segmentation/backend/evaluate_vlm_a4c13.py)
- 数据准备：
  [`backend/scripts/prepare_cvat_segmentation.py`](/Users/mico/Documents/za/git/segmentation/backend/scripts/prepare_cvat_segmentation.py)
  [`backend/scripts/prepare_prompt_seg_data.py`](/Users/mico/Documents/za/git/segmentation/backend/scripts/prepare_prompt_seg_data.py)

## 说明文档

- 后端部署与启动：
  [docs/前后端部署与服务器启动说明.md](/Users/mico/Documents/za/git/segmentation/docs/前后端部署与服务器启动说明.md)
- 顶层脚本说明：
  [顶层Python文件说明.md](/Users/mico/Documents/za/git/segmentation/顶层Python文件说明.md)
- 后端实验结果：
  [docs/final_experiment_results.md](/Users/mico/Documents/za/git/segmentation/docs/final_experiment_results.md)
