# backend 使用说明

这个文档只保留两种最常用场景：

1. 没有模型文件，从零开始复现当前结果
2. 已经拿到模型文件，直接快速部署

默认服务器系统为 Linux，项目目录为：

```text
/root/autodl-tmp/project/segmentation
```

## 一、没有模型文件，如何从零复现

下面这段命令是从零开始准备环境、准备数据、训练纯视觉基线、训练提示式视觉语言模型、再评估当前结果的完整命令流。

执行前请先确保服务器上已经有：

- 当前仓库代码
- `FOCUS-dataset/`
- 公开数据集 `Fetal Echocardiography Second Trimester`

推荐目录结构：

```text
/root/autodl-tmp/project/segmentation/
├── backend/
├── FOCUS-dataset/
├── prepared/
└── public-datasets/
    └── Fetal Echocardiography Second Trimester/
```

完整命令如下：

```bash
cd /root/autodl-tmp/project/segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install modelscope
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'AI-ModelScope/clip-vit-large-patch14',
    cache_dir='/root/autodl-tmp/modelscope_cache'
)
PY
python3 backend/scripts/prepare_cvat_segmentation.py \
  --preset a4c13_poly \
  --dataset-root FOCUS-dataset \
  --output-root prepared/a4c13 \
  --force
python3 backend/scripts/prepare_prompt_seg_data.py \
  --a4c-root prepared/a4c13 \
  --public-root /root/autodl-tmp/project/segmentation/public-datasets/"Fetal Echocardiography Second Trimester" \
  --output-root prepared/prompt_seg \
  --force
python3 backend/train.py \
  --data-root prepared/a4c13 \
  --experiment-dir experiments/a4c13_mobileunet_fpn_pretrained_512 \
  --epochs 120 \
  --batch-size 8 \
  --img-size 512 \
  --lr 3e-4 \
  --pretrained-backbone \
  --amp
python3 backend/train_vlm.py \
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
python3 backend/train_vlm.py \
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
python3 backend/evaluate_vlm_a4c13.py \
  --checkpoint experiments/vlm_stage2_proposal_mscope_large_384/best.pt \
  --data-root prepared/a4c13 \
  --split testing \
  --text-encoder /root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 \
  --img-size 384 \
  --output experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_testing.json
```

跑完后，最关键的结果文件是：

- `experiments/a4c13_mobileunet_fpn_pretrained_512/best.pt`
- `experiments/vlm_stage1_proposal_mscope_large_384/best.pt`
- `experiments/vlm_stage2_proposal_mscope_large_384/best.pt`
- `experiments/vlm_stage2_proposal_mscope_large_384/a4c13_eval_testing.json`

## 二、已经拿到模型文件，如何快速部署

下面这段命令用于“你已经有训练好的 `best.pt`，不想重新训练，只想把当前结果直接跑起来”。

执行前请确保服务器上已经有：

- 当前仓库代码
- 模型文件，例如：
  `/root/autodl-tmp/project/segmentation/server_artifacts/20260321_proposal_best/files/best.pt`

完整命令如下：

```bash
cd /root/autodl-tmp/project/segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install modelscope
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'AI-ModelScope/clip-vit-large-patch14',
    cache_dir='/root/autodl-tmp/modelscope_cache'
)
PY
export SEGMENTATION_MODEL_PATH=/root/autodl-tmp/project/segmentation/server_artifacts/20260321_proposal_best/files/best.pt
export SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
export SEGMENTATION_IMAGE_SIZE=384
export SEGMENTATION_BINARY_THRESHOLD=0.5
export SEGMENTATION_MULTICLASS_THRESHOLD=0.15
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

如果你想让后端常驻运行，直接用下面这段：

```bash
cd /root/autodl-tmp/project/segmentation
screen -S seg_api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install modelscope
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    'AI-ModelScope/clip-vit-large-patch14',
    cache_dir='/root/autodl-tmp/modelscope_cache'
)
PY
export SEGMENTATION_MODEL_PATH=/root/autodl-tmp/project/segmentation/server_artifacts/20260321_proposal_best/files/best.pt
export SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
export SEGMENTATION_IMAGE_SIZE=384
export SEGMENTATION_BINARY_THRESHOLD=0.5
export SEGMENTATION_MULTICLASS_THRESHOLD=0.15
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

启动后可用下面这条检查服务：

```bash
curl http://127.0.0.1:8000/health
```

## 三、本地前端如何连这个后端

服务器后端起来后，你本地还需要开一个端口转发：

```bash
ssh -p <服务器端口> -L 18000:127.0.0.1:8000 root@<服务器地址>
```

然后本地启动前端：

```bash
cd frontend/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

前端里后端地址填：

```text
http://127.0.0.1:18000
```
