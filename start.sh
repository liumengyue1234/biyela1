#!/bin/bash

# 松材线虫病检测系统 - 快速启动脚本

echo "=========================================="
echo "松材线虫病检测系统 - 启动脚本"
echo "=========================================="

# 设置工作目录
PROJECT_DIR="D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection"
cd "$PROJECT_DIR"

echo ""
echo "1. 检查Python环境..."
python --version

echo ""
echo "2. 检查Node.js环境..."
node --version

echo ""
echo "3. 检查MySQL数据库连接..."
echo "请确保MySQL服务正在运行，数据库名称: pine_nematode_db"

echo ""
echo "4. 启动后端服务..."
echo "   后端将在 http://localhost:8080 运行"
gnome-terminal -- bash -c "cd backend && mvn spring-boot:run" &

echo ""
echo "5. 启动前端服务..."
echo "   前端将在 http://localhost:3000 运行"
gnome-terminal -- bash -c "cd frontend && npm install && npm run dev" &

echo ""
echo "=========================================="
echo "服务启动中..."
echo "后端: http://localhost:8080/api"
echo "前端: http://localhost:3000"
echo "=========================================="
echo ""
echo "开始使用系统吧！"
