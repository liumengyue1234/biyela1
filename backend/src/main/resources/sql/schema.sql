-- 松材线虫病检测系统 - 数据库初始化脚本
-- 创建时间: 2024

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS pine_nematode_db
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE pine_nematode_db;

-- 创建检测记录表
CREATE TABLE IF NOT EXISTS detection_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    image_name VARCHAR(255) NOT NULL COMMENT '图像名称',
    image_path VARCHAR(500) NOT NULL COMMENT '图像存储路径',
    result_image_path VARCHAR(500) COMMENT '结果图像路径',
    model_type VARCHAR(50) NOT NULL COMMENT '模型类型: UNET/MASK_RCNN/YOLO',
    nematode_count INT DEFAULT 0 COMMENT '检测到的线虫数量',
    confidence DECIMAL(5,4) DEFAULT 0 COMMENT '置信度',
    processing_time DECIMAL(10,3) COMMENT '处理时间(秒)',
    threshold DECIMAL(3,2) DEFAULT 0.5 COMMENT '置信度阈值',
    result_json TEXT COMMENT '完整结果JSON',
    status VARCHAR(20) DEFAULT 'SUCCESS' COMMENT '状态: SUCCESS/FAILED/PENDING',
    error_message VARCHAR(500) COMMENT '错误信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_model_type (model_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='检测记录表';

-- 创建训练记录表
CREATE TABLE IF NOT EXISTS training_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    model_type VARCHAR(50) NOT NULL COMMENT '模型类型',
    dataset_name VARCHAR(255) COMMENT '数据集名称',
    initial_epoch INT DEFAULT 0 COMMENT '初始轮次',
    final_epoch INT DEFAULT 0 COMMENT '最终轮次',
    initial_loss DECIMAL(10,6) COMMENT '初始损失',
    final_loss DECIMAL(10,6) COMMENT '最终损失',
    initial_accuracy DECIMAL(5,4) COMMENT '初始准确率',
    final_accuracy DECIMAL(5,4) COMMENT '最终准确率',
    best_epoch INT COMMENT '最佳轮次',
    best_metrics TEXT COMMENT '最佳指标JSON',
    model_path VARCHAR(500) COMMENT '模型权重路径',
    training_config TEXT COMMENT '训练配置JSON',
    status VARCHAR(20) DEFAULT 'PENDING' COMMENT '状态',
    error_message VARCHAR(500) COMMENT '错误信息',
    started_at TIMESTAMP NULL COMMENT '开始时间',
    completed_at TIMESTAMP NULL COMMENT '完成时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_type (model_type),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练记录表';

-- 创建模型性能对比表
CREATE TABLE IF NOT EXISTS model_comparison (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL COMMENT '模型类型',
    test_accuracy DECIMAL(5,4) COMMENT '测试准确率',
    mean_iou DECIMAL(5,4) COMMENT 'Mean IoU',
    f1_score DECIMAL(5,4) COMMENT 'F1 Score',
    precision_score DECIMAL(5,4) COMMENT '精确率',
    recall_score DECIMAL(5,4) COMMENT '召回率',
    avg_processing_time DECIMAL(10,3) COMMENT '平均处理时间',
    dataset_size INT COMMENT '测试集大小',
    evaluation_date DATE COMMENT '评估日期',
    notes TEXT COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_model_type_date (model_type, evaluation_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型性能对比表';

-- 插入初始模型性能数据
INSERT INTO model_comparison (model_type, test_accuracy, mean_iou, f1_score, precision_score, recall_score, avg_processing_time, dataset_size, evaluation_date, notes) VALUES
('UNET', 0.8950, 0.8230, 0.8780, 0.8920, 0.8650, 0.245, 200, CURDATE(), '改进U-Net + CBAM注意力机制'),
('MASK_RCNN', 0.8730, 0.7980, 0.8520, 0.8680, 0.8370, 0.523, 200, CURDATE(), 'Mask R-CNN实例分割'),
('YOLO', 0.8580, 0.7650, 0.8310, 0.8450, 0.8180, 0.098, 200, CURDATE(), 'YOLOv8目标检测');
