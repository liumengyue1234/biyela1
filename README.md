# 松材线虫病检测系统

基于深度学习的CT图像松材线虫病自动检测系统

## 项目概述

本项目是一个完整的CT图像松材线虫病检测系统，采用Vue3前端+SpringBoot后端架构，集成了三种先进的深度学习模型：

- **改进U-Net**：结合CBAM注意力机制，分割精度高
- **Mask R-CNN**：实例分割，可区分不同目标
- **YOLO**：实时目标检测，速度快

## 项目结构

```
pine-nematode-detection/
├── backend/                 # SpringBoot后端
│   ├── src/
│   │   └── main/
│   │       ├── java/com/bio/detect/
│   │       │   ├── controller/    # 控制器
│   │       │   ├── service/       # 服务层
│   │       │   ├── model/         # 数据模型
│   │       │   ├── repository/     # 数据访问
│   │       │   └── config/        # 配置类
│   │       └── resources/
│   │           └── application.yml
│   └── pom.xml
│
├── frontend/                # Vue3前端
│   ├── src/
│   │   ├── views/           # 页面组件
│   │   ├── components/      # 公共组件
│   │   ├── api/              # API接口
│   │   ├── router/           # 路由配置
│   │   └── assets/           # 静态资源
│   └── package.json
│
├── models/                  # 深度学习模型
│   ├── scripts/
│   │   ├── data_preprocessing.py   # 数据预处理
│   │   ├── unet_model.py          # U-Net模型
│   │   ├── mask_rcnn_model.py      # Mask R-CNN模型
│   │   ├── yolo_model.py          # YOLO模型
│   │   ├── train_unet.py          # U-Net训练
│   │   ├── train_mask_rcnn.py      # Mask R-CNN训练
│   │   ├── train_yolo.py          # YOLO训练
│   │   ├── detect_unet.py          # U-Net检测
│   │   └── evaluate_models.py      # 模型评估
│   ├── checkpoints/               # 模型权重
│   ├── data/                     # 数据集
│   └── results/                  # 训练结果
│
└── docs/                      # 文档
```

## 技术栈

### 前端
- Vue 3
- Vue Router 4
- Pinia
- Element Plus
- ECharts

### 后端
- Spring Boot 2.7
- Spring Data JPA
- MySQL 8.0
- Lombok

### 深度学习
- PyTorch
- torchvision
- OpenCV
- scikit-learn

## 快速开始

### 1. 环境准备

```bash
# Python环境 (>=3.8)
pip install torch torchvision opencv-python numpy scikit-learn tqdm

# Node.js环境 (>=16)
npm install -g @vue/cli
```

### 2. 数据库配置

创建MySQL数据库：
```sql
CREATE DATABASE pine_nematode_db CHARACTER SET utf8mb4;
```

修改后端配置 `backend/src/main/resources/application.yml`：
```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/pine_nematode_db?useUnicode=true&characterEncoding=utf8
    username: root
    password: your_password
```

### 3. 启动后端

```bash
cd backend
mvn spring-boot:run
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 5. 模型训练

```bash
cd models/scripts

# 数据预处理
python data_preprocessing.py

# 训练U-Net模型
python train_unet.py --epochs 50 --batch_size 8

# 训练Mask R-CNN模型
python train_mask_rcnn.py --epochs 50 --batch_size 4

# 训练YOLO模型
python train_yolo.py --epochs 100 --batch_size 16
```

### 6. 图像检测

```bash
cd models/scripts
python detect_unet.py --image path/to/image.png --model UNET --weights path/to/model.pth
```

## 使用说明

### 图像检测

1. 访问 `http://localhost:3000`
2. 点击"图像检测"菜单
3. 上传CT图像
4. 选择检测模型和参数
5. 点击"开始检测"
6. 查看检测结果

### 模型训练

1. 访问 "模型训练" 页面
2. 选择要训练的模型
3. 配置训练参数
4. 点击"开始训练"
5. 监控训练进度

### 模型对比

1. 访问 "模型对比" 页面
2. 查看各模型的性能指标
3. 选择最优模型进行检测

## 模型性能

| 模型 | 测试准确率 | 平均IoU | F1分数 | 推理时间 |
|------|-----------|--------|--------|----------|
| 改进U-Net | 89.5% | 82.3% | 0.878 | 36ms |
| Mask R-CNN | 87.3% | 79.8% | 0.852 | 54ms |
| YOLO | 85.8% | 76.5% | 0.831 | 24ms |

## API接口

### 检测接口

- `POST /api/detection/upload` - 上传图像并检测
- `POST /api/detection/detect` - 使用指定路径检测
- `GET /api/detection/records` - 获取检测记录
- `GET /api/detection/models` - 获取可用模型

### 训练接口

- `POST /api/training/start` - 开始训练
- `GET /api/training/records` - 获取训练记录
- `GET /api/training/history/{modelType}` - 获取训练历史
- `GET /api/training/comparison` - 模型对比数据

## 注意事项

1. 确保GPU显存 >= 4GB 以进行模型训练
2. 数据集图像建议使用PNG格式
3. 检测阈值建议设置为0.5-0.7之间

## License

MIT License
