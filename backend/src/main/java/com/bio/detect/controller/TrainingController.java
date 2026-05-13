package com.bio.detect.controller;

import com.bio.detect.model.TrainingRecord;
import com.bio.detect.service.TrainingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 训练接口控制器
 */
@Slf4j
@RestController
@RequestMapping("/training")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class TrainingController {
    
    private final TrainingService trainingService;
    
    /**
     * 开始训练
     */
    @PostMapping("/start")
    public ResponseEntity<Map<String, Object>> startTraining(
            @RequestParam String modelType,
            @RequestParam String trainDataset,
            @RequestParam(required = false) String testDataset,
            @RequestParam(defaultValue = "50") Integer epochs,
            @RequestParam(defaultValue = "8") Integer batchSize,
            @RequestParam(defaultValue = "0.001") Double learningRate,
            @RequestParam(required = false) MultipartFile configFile) {
        
        log.info("收到训练请求 - 模型: {}, 轮数: {}, 批大小: {}", modelType, epochs, batchSize);
        
        TrainingRecord record = trainingService.startTraining(
                modelType, trainDataset, testDataset, epochs, batchSize, learningRate, configFile);
        
        Map<String, Object> result = new HashMap<>();
        result.put("recordId", record.getId());
        result.put("status", record.getStatus());
        result.put("message", "训练已启动");
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取训练记录列表
     */
    @GetMapping("/records")
    public ResponseEntity<Map<String, Object>> getRecords(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        List<TrainingRecord> records = trainingService.getTrainingRecords(page, size);
        Map<String, Object> result = new HashMap<>();
        result.put("records", records);
        result.put("total", records.size());
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取指定模型的训练历史
     */
    @GetMapping("/history/{modelType}")
    public ResponseEntity<List<TrainingRecord>> getTrainingHistory(@PathVariable String modelType) {
        List<TrainingRecord> history = trainingService.getTrainingHistory(modelType);
        return ResponseEntity.ok(history);
    }
    
    /**
     * 获取训练记录详情
     */
    @GetMapping("/records/{id}")
    public ResponseEntity<TrainingRecord> getRecord(@PathVariable Long id) {
        TrainingRecord record = trainingService.getTrainingRecord(id);
        if (record == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(record);
    }
    
    /**
     * 获取所有可用模型
     */
    @GetMapping("/models")
    public ResponseEntity<Map<String, String>> getAvailableModels() {
        Map<String, String> models = new HashMap<>();
        models.put("UNET", "改进U-Net分割模型");
        models.put("MASK_RCNN", "Mask R-CNN实例分割模型");
        models.put("YOLO", "YOLO目标检测模型");
        
        return ResponseEntity.ok(models);
    }
    
    /**
     * 模型对比接口
     */
    @GetMapping("/comparison")
    public ResponseEntity<Map<String, Object>> getModelComparison() {
        List<TrainingRecord> unetHistory = trainingService.getTrainingHistory("UNET");
        List<TrainingRecord> maskrcnnHistory = trainingService.getTrainingHistory("MASK_RCNN");
        List<TrainingRecord> yoloHistory = trainingService.getTrainingHistory("YOLO");
        
        Map<String, Object> comparison = new HashMap<>();
        comparison.put("UNET", getLatestMetrics(unetHistory));
        comparison.put("MASK_RCNN", getLatestMetrics(maskrcnnHistory));
        comparison.put("YOLO", getLatestMetrics(yoloHistory));
        
        return ResponseEntity.ok(comparison);
    }
    
    private Map<String, Object> getLatestMetrics(List<TrainingRecord> records) {
        Map<String, Object> metrics = new HashMap<>();
        if (!records.isEmpty()) {
            TrainingRecord latest = records.get(0);
            metrics.put("testAccuracy", latest.getTestAccuracy());
            metrics.put("meanIou", latest.getMeanIou());
            metrics.put("f1Score", latest.getF1Score());
            metrics.put("precision", latest.getPrecisionScore());
            metrics.put("recall", latest.getRecallScore());
            metrics.put("trainingTime", latest.getTrainingTime());
        }
        return metrics;
    }
}
