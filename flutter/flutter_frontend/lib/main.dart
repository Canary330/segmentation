// 导入Flutter核心UI库
import 'package:flutter/material.dart';
// 导入网络请求库（刚装的http包）
import 'package:http/http.dart' as http;
// 导入Flutter自带的JSON解析库（后端返回的是JSON格式）
import 'dart:convert';

// Flutter程序入口，固定写法
void main() {
  runApp(const MyApp());
}

// 根组件，固定写法，负责全局主题
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter+FastAPI', // 页面标题
      theme: ThemeData(primarySwatch: Colors.blue), // 主题色（蓝色）
      home: const HelloPage(), // 主页面：显示后端返回的Hello World
    );//
  }
}

// 主页面组件（核心：调用后端接口+显示数据）
class HelloPage extends StatefulWidget {
  const HelloPage({super.key});

  @override
  State<HelloPage> createState() => _HelloPageState();
}

class _HelloPageState extends State<HelloPage> {
  // 存储后端返回的消息，初始值为「正在请求后端...」
  String _backendMessage = "正在请求后端...";

  // 页面初始化时自动执行：调用FastAPI接口
  @override
  void initState() {
    super.initState();
    _fetchHelloFromFastAPI();
  }

  // 核心方法：调用FastAPI的/hello接口
  Future<void> _fetchHelloFromFastAPI() async {
    try {
      final response = await http.get(
        Uri.parse('http://192.168.2.96:8000/hello'), 
      );

      // 接口请求成功（状态码200，代表后端正常响应）
      if (response.statusCode == 200) {
        // 解析后端返回的JSON数据，转成Flutter能识别的格式
        final Map<String, dynamic> data = json.decode(response.body);
        // 更新页面显示的消息（setState刷新UI）
        setState(() {
          _backendMessage = data['message']; // 取后端返回的message字段
        });
      } else {
        // 接口请求失败（比如端口错了），显示错误码
        setState(() {
          _backendMessage = "请求失败，错误码：${response.statusCode}";
        });
      }
    } catch (e) {
      // 网络错误（比如后端没开、IP错了、防火墙拦截），显示错误信息
      setState(() {
        _backendMessage = "网络错误：$e";
      });
    }
  }

  // 构建页面UI，显示后端返回的消息
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter调用FastAPI'), // 顶部导航栏标题
      ),
      body: Center(
        // 页面中间显示文字，字体放大、加粗，居中
        child: Text(
          _backendMessage,
          style: const TextStyle(
            fontSize: 20, // 字体大小
            fontWeight: FontWeight.bold, // 加粗
            color: Colors.black87, // 字体颜色
          ),
          textAlign: TextAlign.center, // 文字居中
        ),
      ),
    );
  }
}