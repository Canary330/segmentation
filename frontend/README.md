# frontend 目录说明

这个目录现在只保留前端内容，不再混放后端代码。

当前前端采用 Flutter Web，用来做：

- 后端连通性检查
- 图像上传
- 单提示词分割测试
- `13` 类整体分割测试
- 叠加预览
- 导出 `mask` 和预览图

## 目录结构

```text
frontend/
├── flutter_frontend/
└── README.md
```

## 本地启动前端

前提：

- 服务器上的后端已经启动
- 你已经在本地建立了 SSH 端口转发
- 本地后端地址为 `http://127.0.0.1:18000`

启动命令：

```bash
cd frontend/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

## 本地端口说明

前端默认不是直接连云端 `8000` 端口，而是连本地映射端口：

```text
http://127.0.0.1:18000
```

因此你本地需要先执行：

```bash
ssh -p <服务器端口> -L 18000:127.0.0.1:8000 root@<服务器地址>
```

这个步骤必须保留，否则前端会报“找不到后端”。
