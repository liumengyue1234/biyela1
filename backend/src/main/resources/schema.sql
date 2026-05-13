-- 松材线虫病检测系统数据库初始化脚本

-- 创建数据库
CREATE DATABASE IF NOT EXISTS pine_nematode_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE pine_nematode_db;

-- 检测记录表
CREATE TABLE IF NOT EXISTS detection_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(500),
    result_image_path VARCHAR(500),
    model_type VARCHAR(50),
    nematode_count INT DEFAULT 0,
    confidence DOUBLE DEFAULT 0.0,
    status VARCHAR(20),
    result_json TEXT,
    processing_time BIGINT,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_type (model_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 训练记录表
CREATE TABLE IF NOT EXISTS training_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(200),
    model_path VARCHAR(500),
    train_dataset VARCHAR(200),
    test_dataset VARCHAR(200),
    epochs INT,
    batch_size INT,
    learning_rate DOUBLE,
    train_accuracy DOUBLE,
    val_accuracy DOUBLE,
    test_accuracy DOUBLE,
    mean_iou DOUBLE,
    f1_score DOUBLE,
    precision_score DOUBLE,
    recall_score DOUBLE,
    status VARCHAR(20),
    training_time BIGINT,
    log_path VARCHAR(500),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_type (model_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 插入示例数据
INSERT INTO training_records (model_type, model_name, epochs, batch_size, learning_rate, status) VALUES
('UNET', '改进U-Net分割模型 v1.0', 50, 8, 0.001, 'TRAINING'),
('MASK_RCNN', 'Mask R-CNN实例分割模型 v1.0', 50, 4, 0.001, 'TRAINING'),
('YOLO', 'YOLO目标检测模型 v1.0', 100, 16, 0.001, 'TRAINING');
