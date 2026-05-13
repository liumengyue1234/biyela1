package com.bio.detect.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import javax.persistence.*;
import java.time.LocalDateTime;

/**
 * 检测记录实体类
 */
@Data
@Entity
@Table(name = "detection_records")
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DetectionRecord {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    /**
     * 原始图像路径
     */
    @Column(name = "image_path", length = 500)
    private String imagePath;
    
    /**
     * 检测结果图像路径
     */
    @Column(name = "result_image_path", length = 500)
    private String resultImagePath;
    
    /**
     * 使用的模型类型: UNET, MASK_RCNN, YOLO
     */
    @Column(name = "model_type", length = 50)
    private String modelType;
    
    /**
     * 检测到的线虫数量
     */
    @Column(name = "nematode_count")
    private Integer nematodeCount;
    
    /**
     * 检测置信度
     */
    @Column(name = "confidence")
    private Double confidence;
    
    /**
     * 检测状态: PENDING, PROCESSING, COMPLETED, FAILED
     */
    @Column(name = "status", length = 20)
    private String status;
    
    /**
     * 检测结果JSON
     */
    @Column(name = "result_json", columnDefinition = "TEXT")
    private String resultJson;
    
    /**
     * 处理时间(毫秒)
     */
    @Column(name = "processing_time")
    private Long processingTime;
    
    /**
     * 错误信息
     */
    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;
    
    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
    
    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
