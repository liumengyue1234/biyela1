package com.bio.detect.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import java.util.List;

/**
 * 检测请求DTO
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DetectionRequest {
    
    /**
     * 图像路径或URL
     */
    private String imagePath;
    
    /**
     * 模型类型: UNET, MASK_RCNN, YOLO
     */
    private String modelType;
    
    /**
     * 模型版本/路径
     */
    private String modelVersion;
    
    /**
     * 检测阈值
     */
    private Double threshold;
    
    /**
     * 是否返回可视化结果
     */
    private Boolean returnVisualization;
}

/**
 * 检测响应DTO
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DetectionResponse {
    
    /**
     * 检测记录ID
     */
    private Long recordId;
    
    /**
     * 检测状态
     */
    private String status;
    
    /**
     * 检测到的线虫数量
     */
    private Integer nematodeCount;
    
    /**
     * 检测置信度
     */
    private Double confidence;
    
    /**
     * 检测结果详情
     */
    private List<DetectionResult> results;
    
    /**
     * 结果图像URL
     */
    private String resultImageUrl;
    
    /**
     * 处理时间(ms)
     */
    private Long processingTime;
    
    /**
     * 错误信息
     */
    private String errorMessage;
}

/**
 * 单个检测结果
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DetectionResult {
    
    /**
     * 边界框坐标 [x1, y1, x2, y2]
     */
    private List<Double> bbox;
    
    /**
     * 置信度
     */
    private Double score;
    
    /**
     * 类别
     */
    private String label;
    
    /**
     * 分割掩码路径(针对分割模型)
     */
    private String maskPath;
}
