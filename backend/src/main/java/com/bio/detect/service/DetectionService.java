package com.bio.detect.service;

import com.bio.detect.model.DTO;
import com.bio.detect.model.DetectionRecord;
import com.bio.detect.repository.DetectionRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * 检测服务类
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DetectionService {
    
    private final DetectionRecordRepository detectionRecordRepository;
    
    @Value("${file.upload.path}")
    private String uploadPath;
    
    @Value("${python.executable}")
    private String pythonPath;
    
    @Value("${python.script-path}")
    private String scriptPath;
    
    /**
     * 上传图像并进行检测
     */
    public DTO.DetectionResponse uploadAndDetect(MultipartFile file, String modelType, Double threshold) {
        String imagePath = null;
        DetectionRecord record = null;
        
        try {
            // 保存上传的文件
            imagePath = saveUploadFile(file);
            log.info("文件上传成功: {}", imagePath);
            
            // 创建检测记录
            record = DetectionRecord.builder()
                    .imagePath(imagePath)
                    .modelType(modelType)
                    .status("PENDING")
                    .confidence(0.0)
                    .nematodeCount(0)
                    .build();
            record = detectionRecordRepository.save(record);
            
            // 执行检测
            return executeDetection(record, modelType, threshold);
            
        } catch (IOException e) {
            log.error("文件上传失败", e);
            return DTO.DetectionResponse.builder()
                    .status("FAILED")
                    .errorMessage("文件上传失败: " + e.getMessage())
                    .build();
        }
    }
    
    /**
     * 使用指定图像路径进行检测
     */
    public DTO.DetectionResponse detect(String imagePath, String modelType, Double threshold) {
        DetectionRecord record = DetectionRecord.builder()
                .imagePath(imagePath)
                .modelType(modelType)
                .status("PENDING")
                .confidence(0.0)
                .nematodeCount(0)
                .build();
        record = detectionRecordRepository.save(record);
        
        return executeDetection(record, modelType, threshold);
    }
    
    /**
     * 执行检测
     */
    @Async
    public DTO.DetectionResponse executeDetection(DetectionRecord record, String modelType, Double threshold) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 更新状态为处理中
            record.setStatus("PROCESSING");
            detectionRecordRepository.save(record);
            
            // 调用Python脚本执行检测
            String resultJson = callPythonDetection(record.getImagePath(), modelType, threshold);
            
            // 解析结果
            DTO.DetectionResponse response = parseDetectionResult(resultJson, record);
            response.setRecordId(record.getId());
            response.setProcessingTime(System.currentTimeMillis() - startTime);
            
            // 更新记录
            record.setStatus("COMPLETED");
            record.setNematodeCount(response.getNematodeCount());
            record.setConfidence(response.getConfidence());
            record.setResultJson(resultJson);
            record.setProcessingTime(response.getProcessingTime());
            detectionRecordRepository.save(record);
            
            return response;
            
        } catch (Exception e) {
            log.error("检测执行失败", e);
            record.setStatus("FAILED");
            record.setErrorMessage(e.getMessage());
            detectionRecordRepository.save(record);
            
            return DTO.DetectionResponse.builder()
                    .recordId(record.getId())
                    .status("FAILED")
                    .errorMessage(e.getMessage())
                    .build();
        }
    }
    
    /**
     * 调用Python检测脚本
     */
    private String callPythonDetection(String imagePath, String modelType, Double threshold) throws IOException {
        String script = scriptPath + "/detect_" + modelType.toLowerCase() + ".py";
        
        ProcessBuilder pb = new ProcessBuilder(
                pythonPath,
                script,
                "--image", imagePath,
                "--threshold", String.valueOf(threshold != null ? threshold : 0.5)
        );
        pb.redirectErrorStream(true);
        
        Process process = pb.start();
        
        // 读取输出
        StringBuilder output = new StringBuilder();
        try (java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
        }
        
        try {
            process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        return output.toString();
    }
    
    /**
     * 解析检测结果
     */
    private DTO.DetectionResponse parseDetectionResult(String resultJson, DetectionRecord record) {
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            DTO.DetectionResponse response = mapper.readValue(resultJson, DTO.DetectionResponse.class);
            response.setResultImageUrl("/results/" + record.getId() + ".png");
            return response;
        } catch (Exception e) {
            return DTO.DetectionResponse.builder()
                    .status("FAILED")
                    .errorMessage("结果解析失败: " + e.getMessage())
                    .build();
        }
    }
    
    /**
     * 保存上传文件
     */
    private String saveUploadFile(MultipartFile file) throws IOException {
        Path uploadDir = Paths.get(uploadPath);
        if (!Files.exists(uploadDir)) {
            Files.createDirectories(uploadDir);
        }
        
        String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
        Path filePath = uploadDir.resolve(filename);
        Files.copy(file.getInputStream(), filePath);
        
        return filePath.toString();
    }
    
    /**
     * 获取检测记录列表
     */
    public List<DetectionRecord> getDetectionRecords(int page, int size) {
        return detectionRecordRepository.findAll(
                org.springframework.data.domain.PageRequest.of(page, size,
                        org.springframework.data.domain.Sort.by("createdAt").descending()))
                .getContent();
    }
    
    /**
     * 获取检测记录详情
     */
    public DetectionRecord getDetectionRecord(Long id) {
        return detectionRecordRepository.findById(id).orElse(null);
    }
}
