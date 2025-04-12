# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖列表
COPY requirements.txt .

# 安装依赖（使用清华镜像加速）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 运行命令（使用Gunicorn）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:create_app()"]