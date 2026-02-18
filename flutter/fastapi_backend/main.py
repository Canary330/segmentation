from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建FastAPI应用
app = FastAPI()

# 配置跨域（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义/hello接口，返回Hello World
@app.get("/hello")
async def hello_world():
    return {"message": "Hello World! This is FastAPI Backend Data"}