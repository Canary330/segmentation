# Flutter 最终测试说明

这个目录包含两部分：

- `flutter_frontend`：本地运行的 Flutter 测试前端
- `fastapi_backend`：云端运行的推理后端

## 一、先在云端启动后端

进入云端项目目录后，执行：

```bash
cd /root/autodl-tmp/project/segmentation
pip install -r flutter/fastapi_backend/requirements.txt.txt
export SEGMENTATION_MODEL_PATH=/root/autodl-tmp/project/segmentation/experiments/vlm_stage2_proposal_mscope_large_384/best.pt
export SEGMENTATION_TEXT_ENCODER=/root/autodl-tmp/modelscope_cache/AI-ModelScope/clip-vit-large-patch14
export SEGMENTATION_IMAGE_SIZE=384
export SEGMENTATION_BINARY_THRESHOLD=0.5
export SEGMENTATION_MULTICLASS_THRESHOLD=0.15
uvicorn flutter.fastapi_backend.main:app --host 0.0.0.0 --port 8000
```

如果希望后端常驻运行，建议放进 `screen`。

## 二、本地建立到云端的端口转发

在本地终端执行：

```bash
ssh -p 50917 -L 18000:127.0.0.1:8000 root@connect.westc.seetacloud.com
```

建立成功后，本地地址 `http://127.0.0.1:18000` 就会映射到云端后端。

## 三、运行 Flutter 前端

进入前端目录：

```bash
cd flutter/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

## 四、前端支持的测试方式

- 检查后端状态
- 选择本地图像
- 输入单个提示词进行二值分割
- 直接执行 13 类整体分割

## 五、建议测试流程

1. 先点“检查后端”，确认连接成功
2. 选择一张测试图像
3. 输入一个结构名称，例如“左心房”
4. 先做“单提示词分割”
5. 再做“13 类整体分割”

如果页面提示网络错误，优先检查两件事：

- 云端 `uvicorn` 是否真的在运行
- 本地 `ssh -L` 隧道是否还保持连接
