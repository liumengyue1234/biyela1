package com.bio.detect.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.hibernate.annotations.CreationTimestamp;
import javax.persistence.*;
import java.time.LocalDateTime;

/**
 * 训练记录实体类
 */
@Data
@Entity
@Table(name = "training_records")
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrainingRecord {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    /**
     * 模型类型: UNET, MASK_RCNN, YOLO
     */
    @Column(name = "model_type", length = 50, nullable = false)
    private String modelType;
    
    /**
     * 模型名称
     */
    @Column(name = "model_name", length = 200)
    private String modelName;
    
    /**
     * 模型保存路径
     */
    @Column(name = "model_path", length = 500)
    private String modelPath;
    
    /**
     * 训练数据集
     */
    @Column(name = "train_dataset", length = 200)
    private String trainDataset;
    
    /**
     * 测试数据集
     */
    @Column(name = "test_dataset", length = 200)
    private String testDataset;
    
    /**
     * 训练轮数
     */
    @Column(name = "epochs")
    private Integer epochs;
    
    /**
     * 批大小
     */
    @Column(name = "batch_size")
    private Integer batchSize;
    
    /**
     * 学习率
     */
    @Column(name = "learning_rate")
    private Double learningRate;
    
    /**
     * 训练集准确率
     */
    @Column(name = "train_accuracy")
    private Double trainAccuracy;
    
    /**
     * 验证集准确率
     */
    @Column(name = "val_accuracy")
    private Double valAccuracy;
    
    /**
     * 测试集准确率
     */
    @Column(name = "test_accuracy")
    private Double testAccuracy;
    
    /**
     * 平均IoU
     */
    @Column(name = "mean_iou")
    private Double meanIou;
    
    /**
     * F1分数
     */
    @Column(name = "f1_score")
    private Double f1Score;
    
    /**
     * 精确率
     */
    @Column(name = "precision_score")
    private Double precisionScore;
    
    /**
     * 召回率
     */
    @Column(name = "recall_score")
    private Double recallScore;
    
    /**
     * 训练状态: TRAINING, COMPLETED, FAILED
     */
    @Column(name = "status", length = 20)
    private String status;
    
    /**
     * 训练耗时(秒)
     */
    @Column(name = "training_time")
    private Long trainingTime;
    
    /**
     * 训练日志路径
     */
    @Column(name = "log_path", length = 500)
    private String logPath;
    
    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
