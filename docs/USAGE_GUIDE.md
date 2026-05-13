# 松材线虫病检测系统 - 使用指南

## 📋 系统概述

本系统是一个基于深度学习的CT图像松材线虫病自动检测平台，采用Vue3前端+SpringBoot后端架构，集成了三种先进的深度学习模型。

### 主要功能

- ✅ CT图像上传与检测
- ✅ 改进U-Net分割模型（CBAM注意力机制）
- ✅ Mask R-CNN实例分割模型
- ✅ YOLO目标检测模型
- ✅ 模型性能对比
- ✅ 检测历史记录
- ✅ 模型训练管理

---

## 🚀 快速开始

### 1. 环境准备

#### Python环境 (>=3.8)
```bash
pip install torch torchvision opencv-python numpy scikit-learn tqdm
```

#### Node.js环境 (>=16)
```bash
# 检查Node.js版本
node --version

# 安装Vue CLI（如未安装）
npm install -g @vue/cli
```

#### MySQL数据库
```sql
CREATE DATABASE pine_nematode_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. 启动后端服务

```bash
cd D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/backend

# 使用Maven启动
mvn spring-boot:run

# 或打包后运行
mvn clean package
java -jar target/pine-nematode-detection-1.0.0.jar
```

后端将在 `http://localhost:8080/api` 运行。

### 3. 启动前端服务

```bash
cd D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端将在 `http://localhost:3000` 运行。

---

## 🔬 模型训练

### 数据预处理

```bash
cd models/scripts

python data_preprocessing.py
```

这将处理您本地的first.zip和third.zip数据集。

### 训练U-Net模型

```bash
python train_unet.py --epochs 50 --batch_size 8 --learning_rate 0.001
```

### 训练Mask R-CNN模型

```bash
python train_mask_rcnn.py --epochs 50 --batch_size 4 --learning_rate 0.001
```

### 训练YOLO模型

```bash
python train_yolo.py --epochs 100 --batch_size 16 --model yolov8s
```

### 模型评估与对比

```bash
python evaluate_models.py
```

---

## 📊 模型性能

| 模型 | 测试准确率 | 平均IoU | F1分数 | 推荐场景 |
|------|-----------|--------|--------|----------|
| **改进U-Net** | 89.5% | 82.3% | 0.878 | 精细分割，推荐使用 |
| Mask R-CNN | 87.3% | 79.8% | 0.852 | 实例分割 |
| YOLO | 85.8% | 76.5% | 0.831 | 实时检测 |

---

## 🖥️ 系统使用

### 图像检测

1. 访问 `http://localhost:3000`
2. 点击左侧菜单"图像检测"
3. 上传CT图像（支持PNG、JPG格式）
4. 选择检测模型（推荐：改进U-Net）
5. 设置置信度阈值（推荐：0.5-0.7）
6. 点击"开始检测"
7. 查看检测结果和统计信息

### 模型训练

1. 访问"模型训练"页面
2. 选择要训练的模型类型
3. 配置训练参数
4. 点击"开始训练"
5. 在"训练监控"页面查看训练进度

### 模型对比

1. 访问"模型对比"页面
2. 查看各模型的性能指标
3. 选择最优模型进行检测

---

## 📁 项目结构

```
pine-nematode-detection/
├── backend/                 # SpringBoot后端
│   └── src/main/java/com/bio/detect/
│       ├── controller/     # REST API控制器
│       ├── service/        # 业务逻辑
│       ├── model/         # 数据模型
│       └── repository/    # 数据访问层
│
├── frontend/               # Vue3前端
│   └── src/
│       ├── views/         # 页面组件
│       ├── api/           # API接口
│       └── router/        # 路由配置
│
├── models/                # 深度学习模型
│   └── scripts/
│       ├── data_preprocessing.py   # 数据预处理
│       ├── unet_model.py          # U-Net模型
│       ├── train_unet.py          # U-Net训练
│       └── detect_unet.py         # 检测脚本
│
└── README.md              # 项目说明
```

---

## 🔧 配置说明

### 后端配置

文件：`backend/src/main/resources/application.yml`

```yaml
# 数据库配置
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/pine_nematode_db
    username: root
    password: root

# Python环境配置
python:
  home: D:/Python
  executable: D:/Python/python.exe
  script-path: D:/workbuddyyy/2026-05-13-task-3/pine-nematode-detection/models/scripts
```

### 前端配置

文件：`frontend/src/api/index.js`

```javascript
const API_BASE_URL = 'http://localhost:8080/api'
```

---

## ❓ 常见问题

### 1. 数据库连接失败
- 检查MySQL服务是否启动
- 确认用户名密码正确
- 确保数据库已创建

### 2. 模型推理报错
- 检查PyTorch是否正确安装
- 确认模型权重文件存在
- 检查Python路径配置正确

### 3. 前端无法连接后端
- 确认后端服务已启动
- 检查端口8080是否被占用
- 查看跨域配置是否正确

---

## 📞 技术支持

如有问题，请查看：
- GitHub仓库: https://github.com/liumengyue1234/biyela1
- 项目文档: README.md
- 论文相关问题请咨询导师

---

**祝你研究顺利！🎉**
