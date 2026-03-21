import 'dart:convert';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:universal_html/html.dart' as html;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '胎儿超声分割测试台',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0E7490)),
        useMaterial3: true,
      ),
      home: const SegmentationDemoPage(),
    );
  }
}

class SegmentationDemoPage extends StatefulWidget {
  const SegmentationDemoPage({super.key});

  @override
  State<SegmentationDemoPage> createState() => _SegmentationDemoPageState();
}

class _SegmentationDemoPageState extends State<SegmentationDemoPage> {
  static const List<String> _defaultA4c13Labels = <String>[
    'background',
    'DAO',
    'LA',
    'RA',
    'LV',
    'RV',
    'VS',
    'IS',
    'SP',
    'RB',
    'LVW',
    'RVW',
    'LL',
    'RL',
  ];

  static const List<Color> _classColors = <Color>[
    Color(0x00000000),
    Color(0xFF0066FF),
    Color(0xFF00B0F0),
    Color(0xFF00C853),
    Color(0xFFFFC107),
    Color(0xFFFF5722),
    Color(0xFF9C27B0),
    Color(0xFFE91E63),
    Color(0xFF795548),
    Color(0xFF4CAF50),
    Color(0xFF3F51B5),
    Color(0xFFFF9800),
    Color(0xFF607D8B),
    Color(0xFFF44336),
  ];

  static const String _defaultBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://127.0.0.1:18000',
  );

  late final TextEditingController _baseUrlController;
  late final TextEditingController _promptController;

  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  Uint8List? _resultMaskBytes;
  Uint8List? _displayMaskBytes;
  Uint8List? _overlayPreviewBytes;
  Uint8List? _overlayMaskLayerBytes;
  String _statusText = '请先连接后端，再选择图像进行测试。';
  String? _lastPromptUsed;
  bool _isMulticlassPrediction = false;
  bool _busy = false;
  List<String> _labels = const [];
  int? _foregroundPixels;

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController(text: _defaultBaseUrl);
    _promptController = TextEditingController(text: '左心房');
    _checkHealth();
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _promptController.dispose();
    super.dispose();
  }

  String get _baseUrl => _baseUrlController.text.trim().replaceAll(RegExp(r'/$'), '');

  Future<void> _checkHealth() async {
    setState(() {
      _busy = true;
      _statusText = '正在检查后端状态...';
    });
    try {
      final response = await http.get(Uri.parse('$_baseUrl/health'));
      if (response.statusCode != 200) {
        throw Exception('状态码 ${response.statusCode}');
      }
      final data = json.decode(response.body) as Map<String, dynamic>;
      setState(() {
        _statusText =
            '后端连接成功：status=${data['status']}，模型已加载=${data['model_loaded']}，已配置权重=${data['checkpoint_configured']}';
      });
    } catch (error) {
      setState(() {
        _statusText = '后端连接失败：$error';
      });
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['png', 'jpg', 'jpeg', 'bmp'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) {
      return;
    }
    final file = result.files.single;
    if (file.bytes == null) {
      setState(() {
        _statusText = '读取图像失败：没有拿到文件内容。';
      });
      return;
    }
    setState(() {
      _selectedImageBytes = file.bytes;
      _selectedImageName = file.name;
      _resultMaskBytes = null;
      _displayMaskBytes = null;
      _overlayPreviewBytes = null;
      _overlayMaskLayerBytes = null;
      _labels = const [];
      _foregroundPixels = null;
      _lastPromptUsed = null;
      _isMulticlassPrediction = false;
      _statusText = '已选择图像：${file.name}';
    });
  }

  Future<void> _predictSinglePrompt() async {
    if (_selectedImageBytes == null) {
      setState(() {
        _statusText = '请先选择图像。';
      });
      return;
    }
    if (_promptController.text.trim().isEmpty) {
      setState(() {
        _statusText = '请输入提示词。';
      });
      return;
    }

    setState(() {
      _busy = true;
      _statusText = '正在执行单提示词分割...';
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/predict'));
      request.fields['prompt'] = _promptController.text.trim();
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          _selectedImageBytes!,
          filename: _selectedImageName ?? 'image.png',
        ),
      );
      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) {
        throw Exception('状态码 ${response.statusCode}，响应：${response.body}');
      }
      final data = json.decode(response.body) as Map<String, dynamic>;
      final maskBytes = base64Decode(data['mask_png_base64'] as String);
      setState(() {
        _resultMaskBytes = maskBytes;
        _displayMaskBytes = maskBytes;
        _foregroundPixels = data['foreground_pixels'] as int?;
        _labels = const [];
        _lastPromptUsed = data['prompt'] as String? ?? _promptController.text.trim();
        _isMulticlassPrediction = false;
        _statusText = '单提示词分割完成：${data['prompt']}';
      });
      await _buildOverlayPreview();
    } catch (error) {
      setState(() {
        _statusText = '单提示词分割失败：$error';
      });
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  Future<void> _predictA4c13() async {
    if (_selectedImageBytes == null) {
      setState(() {
        _statusText = '请先选择图像。';
      });
      return;
    }

    setState(() {
      _busy = true;
      _statusText = '正在执行 13 类整体分割...';
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/predict_13class'));
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          _selectedImageBytes!,
          filename: _selectedImageName ?? 'image.png',
        ),
      );
      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) {
        throw Exception('状态码 ${response.statusCode}，响应：${response.body}');
      }
      final data = json.decode(response.body) as Map<String, dynamic>;
      final maskBytes = base64Decode(data['mask_png_base64'] as String);
      final labels = (data['labels'] as List<dynamic>).cast<String>();
      setState(() {
        _resultMaskBytes = maskBytes;
        _displayMaskBytes = _buildColorizedMulticlassMask(maskBytes);
        _labels = labels;
        _foregroundPixels = null;
        _lastPromptUsed = null;
        _isMulticlassPrediction = true;
        _statusText = '13 类分割完成。';
      });
      await _buildOverlayPreview();
    } catch (error) {
      setState(() {
        _statusText = '13 类分割失败：$error';
      });
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  Widget _buildImageCard({
    required String title,
    required Uint8List? bytes,
    required String emptyText,
  }) {
    return Expanded(
      child: Card(
        clipBehavior: Clip.antiAlias,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF1F5F9),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: bytes == null
                      ? Center(child: Text(emptyText, textAlign: TextAlign.center))
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.memory(bytes, fit: BoxFit.contain),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Uint8List _buildColorizedMulticlassMask(Uint8List rawMaskBytes) {
    final mask = img.decodeImage(rawMaskBytes);
    if (mask == null) {
      return rawMaskBytes;
    }

    final colorized = img.Image(width: mask.width, height: mask.height, numChannels: 4);
    for (var y = 0; y < mask.height; y++) {
      for (var x = 0; x < mask.width; x++) {
        final pixel = mask.getPixel(x, y);
        final classId = img.getLuminance(pixel).round().clamp(0, _classColors.length - 1);
        final color = _classColors[classId];
        colorized.setPixelRgba(x, y, color.red, color.green, color.blue, 255);
      }
    }
    return Uint8List.fromList(img.encodePng(colorized));
  }

  Future<void> _buildOverlayPreview() async {
    if (_selectedImageBytes == null || _resultMaskBytes == null) {
      return;
    }

    final original = img.decodeImage(_selectedImageBytes!);
    final mask = img.decodeImage(_resultMaskBytes!);
    if (original == null || mask == null) {
      return;
    }

    final resizedMask = mask.width == original.width && mask.height == original.height
        ? mask
        : img.copyResize(mask, width: original.width, height: original.height, interpolation: img.Interpolation.nearest);

    final preview = img.Image.from(original);
    final overlayLayer = img.Image(width: original.width, height: original.height, numChannels: 4);
    const double singleFillAlpha = 0.62;
    const double singleEdgeAlpha = 0.9;
    const int singleRed = 0;
    const int singleGreen = 102;
    const int singleBlue = 255;

    for (var y = 0; y < preview.height; y++) {
      for (var x = 0; x < preview.width; x++) {
        final maskPixel = resizedMask.getPixel(x, y);
        final classId = img.getLuminance(maskPixel).round();
        if (classId == 0) {
          overlayLayer.setPixelRgba(x, y, 0, 0, 0, 0);
          continue;
        }

        final currentColor = _isMulticlassPrediction
            ? _classColors[classId.clamp(0, _classColors.length - 1)]
            : const Color(0xFF0066FF);
        final isEdge =
            x == 0 ||
            y == 0 ||
            x == preview.width - 1 ||
            y == preview.height - 1 ||
            img.getLuminance(resizedMask.getPixel(x - 1, y)).round() != classId ||
            img.getLuminance(resizedMask.getPixel(x + 1, y)).round() != classId ||
            img.getLuminance(resizedMask.getPixel(x, y - 1)).round() != classId ||
            img.getLuminance(resizedMask.getPixel(x, y + 1)).round() != classId;
        final alpha = _isMulticlassPrediction
            ? (isEdge ? 0.9 : 0.55)
            : (isEdge ? singleEdgeAlpha : singleFillAlpha);
        final src = preview.getPixel(x, y);
        final r = (src.r * (1 - alpha) + currentColor.red * alpha).round();
        final g = (src.g * (1 - alpha) + currentColor.green * alpha).round();
        final b = (src.b * (1 - alpha) + currentColor.blue * alpha).round();
        preview.setPixelRgba(x, y, r, g, b, src.a.round());
        overlayLayer.setPixelRgba(
          x,
          y,
          currentColor.red,
          currentColor.green,
          currentColor.blue,
          (255 * alpha).round(),
        );
      }
    }

    setState(() {
      _overlayPreviewBytes = Uint8List.fromList(img.encodePng(preview));
      _overlayMaskLayerBytes = Uint8List.fromList(img.encodePng(overlayLayer));
    });
  }

  void _downloadBytes(Uint8List bytes, String filename) {
    final blob = html.Blob([bytes]);
    final url = html.Url.createObjectUrlFromBlob(blob);
    final anchor = html.AnchorElement(href: url)
      ..download = filename
      ..style.display = 'none';
    html.document.body?.children.add(anchor);
    anchor.click();
    anchor.remove();
    html.Url.revokeObjectUrl(url);
  }

  void _showOverlayPreviewDialog() {
    if (_selectedImageBytes == null ||
        _resultMaskBytes == null ||
        _overlayPreviewBytes == null ||
        _overlayMaskLayerBytes == null) {
      setState(() {
        _statusText = '请先完成一次分割，再查看叠加预览。';
      });
      return;
    }

    showDialog<void>(
      context: context,
      builder: (context) {
        return Dialog(
          insetPadding: const EdgeInsets.all(24),
          child: SizedBox(
            width: 980,
            height: 760,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Expanded(
                        child: Text(
                          '叠加预览',
                          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                        ),
                      ),
                      OutlinedButton(
                        onPressed: () => _downloadBytes(
                          _resultMaskBytes!,
                          'segmentation_mask.png',
                        ),
                        child: const Text('导出 Mask'),
                      ),
                      const SizedBox(width: 12),
                      FilledButton(
                        onPressed: () => _downloadBytes(
                          _overlayPreviewBytes!,
                          'segmentation_preview.png',
                        ),
                        child: const Text('导出预览图'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _foregroundPixels == 0
                        ? '当前输出前景像素为 0，说明这次分割结果为空。'
                        : _isMulticlassPrediction
                            ? '显示方式：原图叠加 13 类彩色区域，不同颜色对应不同结构。'
                            : '显示方式：原图叠加半透明蓝色 mask，边界已加重显示。',
                  ),
                  if (!_isMulticlassPrediction && _lastPromptUsed != null) ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: const Color(0xFFE0F2FE),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        '当前结构：$_lastPromptUsed',
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                  if (_isMulticlassPrediction && _labels.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 10,
                      runSpacing: 10,
                      children: [
                        for (var i = 1; i < _labels.length && i < _classColors.length; i++)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(999),
                              border: Border.all(color: const Color(0xFFCBD5E1)),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(
                                  width: 14,
                                  height: 14,
                                  decoration: BoxDecoration(
                                    color: _classColors[i],
                                    borderRadius: BorderRadius.circular(999),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Text(_labels[i], style: const TextStyle(fontWeight: FontWeight.w600)),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ],
                  const SizedBox(height: 12),
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: const Color(0xFFF1F5F9),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.memory(
                              _selectedImageBytes!,
                              fit: BoxFit.contain,
                            ),
                            Image.memory(
                              _overlayMaskLayerBytes!,
                              fit: BoxFit.contain,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('胎儿超声分割最终测试台')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                SizedBox(
                  width: 420,
                  child: TextField(
                    controller: _baseUrlController,
                    decoration: const InputDecoration(
                      labelText: '后端地址',
                      hintText: '例如 http://127.0.0.1:18000',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                FilledButton(
                  onPressed: _busy ? null : _checkHealth,
                  child: const Text('检查后端'),
                ),
                OutlinedButton(
                  onPressed: _busy ? null : _pickImage,
                  child: const Text('选择图像'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    decoration: const InputDecoration(
                      labelText: '提示词',
                      hintText: '例如：左心房、右心室、室间隔',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                FilledButton(
                  onPressed: _busy ? null : _predictSinglePrompt,
                  child: const Text('单提示词分割'),
                ),
                const SizedBox(width: 12),
                FilledButton.tonal(
                  onPressed: _busy ? null : _predictA4c13,
                  child: const Text('13 类整体分割'),
                ),
                const SizedBox(width: 12),
                OutlinedButton(
                  onPressed: _busy ? null : _showOverlayPreviewDialog,
                  child: const Text('查看叠加预览'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Card(
              color: const Color(0xFFF8FAFC),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_statusText),
                    if (_selectedImageName != null) ...[
                      const SizedBox(height: 6),
                      Text('当前图像：$_selectedImageName'),
                    ],
                    if (_foregroundPixels != null) ...[
                      const SizedBox(height: 6),
                      Text('前景像素数：$_foregroundPixels'),
                    ],
                    if (_labels.isNotEmpty) ...[
                      const SizedBox(height: 6),
                      Text('标签顺序：${_labels.join('、')}'),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Row(
                children: [
                  _buildImageCard(
                    title: '输入图像',
                    bytes: _selectedImageBytes,
                    emptyText: '请选择一张测试图像',
                  ),
                  const SizedBox(width: 12),
                  _buildImageCard(
                    title: '模型输出掩码',
                    bytes: _displayMaskBytes ?? _resultMaskBytes,
                    emptyText: '执行分割后会在这里显示结果',
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
