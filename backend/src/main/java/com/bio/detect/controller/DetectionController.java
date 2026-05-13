package com.bio.detect.controller;

import com.bio.detect.model.DTO;
import com.bio.detect.model.DetectionRecord;
import com.bio.detect.service.DetectionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 检测接口控制器
 */
@Slf4j
@RestController
@RequestMapping("/detection")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class DetectionController {
    
    private final DetectionService detectionService;
    
    /**
     * 上传图像并检测
     */
    @PostMapping("/upload")
    public ResponseEntity<DTO.DetectionResponse> uploadAndDetect(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "modelType", defaultValue = "UNET") String modelType,
            @RequestParam(value = "threshold", defaultValue = "0.5") Double threshold) {
        
        log.info("收到检测请求 - 模型: {}, 阈值: {}", modelType, threshold);
        DTO.DetectionResponse response = detectionService.uploadAndDetect(file, modelType, threshold);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * 使用指定图像路径检测
     */
    @PostMapping("/detect")
    public ResponseEntity<DTO.DetectionResponse> detect(
            @RequestBody Map<String, String> request) {
        
        String imagePath = request.get("imagePath");
        String modelType = request.getOrDefault("modelType", "UNET");
        Double threshold = Double.parseDouble(request.getOrDefault("threshold", "0.5"));
        
        log.info("收到检测请求 - 图像: {}, 模型: {}", imagePath, modelType);
        DTO.DetectionResponse response = detectionService.detect(imagePath, modelType, threshold);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * 获取检测记录列表
     */
    @GetMapping("/records")
    public ResponseEntity<Map<String, Object>> getRecords(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        List<DetectionRecord> records = detectionService.getDetectionRecords(page, size);
        Map<String, Object> result = new HashMap<>();
        result.put("records", records);
        result.put("total", records.size());
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * 获取检测记录详情
     */
    @GetMapping("/records/{id}")
    public ResponseEntity<DetectionRecord> getRecord(@PathVariable Long id) {
        DetectionRecord record = detectionService.getDetectionRecord(id);
        if (record == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(record);
    }
    
    /**
     * 获取支持的模型列表
     */
    @GetMapping("/models")
    public ResponseEntity<Map<String, String>> getAvailableModels() {
        Map<String, String> models = new HashMap<>();
        models.put("UNET", "改进U-Net分割模型 - 语义分割，适合精细边界检测");
        models.put("MASK_RCNN", "Mask R-CNN实例分割模型 - 实例分割，可区分不同目标");
        models.put("YOLOv8", "YOLOv8 - Ultralytics经典版本，精度与速度平衡");
        models.put("YOLO11", "YOLO11 - 2024年新版，增强特征提取能力");
        models.put("YOLOv10", "YOLOv10 - 清华大学端到端版本，无需NMS后处理");

        return ResponseEntity.ok(models);
    }
}
