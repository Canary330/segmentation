# Flutter 前端说明

这是当前项目的 Flutter Web 测试前端。

它负责：

- 选择测试图像
- 输入提示词
- 调用后端 `/predict`
- 调用后端 `/predict_13class`
- 显示彩色结果和叠加预览

## 启动前需要确认

1. 云端后端已经启动
2. 本地 SSH 隧道已经建立
3. 本地地址 `http://127.0.0.1:18000` 能映射到云端 `8000`

## 启动命令

```bash
cd frontend/flutter_frontend
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:18000
```

## 前端里应该填写的后端地址

```text
http://127.0.0.1:18000
```

不要直接填云端地址，当前推荐做法是始终通过本地隧道访问。
