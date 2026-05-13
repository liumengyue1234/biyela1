package com.bio.detect.service;

import com.bio.detect.model.TrainingRecord;
import com.bio.detect.repository.TrainingRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;

/**
 * 训练服务类
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class TrainingService {
    
    private final TrainingRecordRepository trainingRecordRepository;
    
    @Value("${python.executable}")
    private String pythonPath;
    
    @Value("${python.script-path}")
    private String scriptPath;
    
    @Value("${file.model.path}")
    private String modelPath;
    
    /**
     * 开始训练
     */
    public TrainingRecord startTraining(
            String modelType,
            String trainDataset,
            String testDataset,
            Integer epochs,
            Integer batchSize,
            Double learningRate,
            MultipartFile configFile) {
        
        TrainingRecord record = TrainingRecord.builder()
                .modelType(modelType)
                .trainDataset(trainDataset)
                .testDataset(testDataset)
                .epochs(epochs)
                .batchSize(batchSize)
                .learningRate(learningRate)
                .status("TRAINING")
                .modelPath(modelPath + "/" + modelType.toLowerCase())
                .build();
        
        record = trainingRecordRepository.save(record);
        
        try {
            // 执行训练
            long startTime = System.currentTimeMillis();
            TrainingResult result = executeTraining(record, configFile);
            long trainingTime = (System.currentTimeMillis() - startTime) / 1000;
            
            // 更新训练结果
            record.setStatus("COMPLETED");
            record.setTrainingTime(trainingTime);
            record.setTrainAccuracy(result.trainAccuracy);
            record.setValAccuracy(result.valAccuracy);
            record.setTestAccuracy(result.testAccuracy);
            record.setMeanIou(result.meanIou);
            record.setPrecisionScore(result.precision);
            record.setRecallScore(result.recall);
            record.setF1Score(result.f1Score);
            record.setLogPath(result.logPath);
            
            return trainingRecordRepository.save(record);
            
        } catch (Exception e) {
            log.error("训练失败", e);
            record.setStatus("FAILED");
            record.setErrorMessage(e.getMessage());
            return trainingRecordRepository.save(record);
        }
    }
    
    /**
     * 执行Python训练脚本
     */
    private TrainingResult executeTraining(TrainingRecord record, MultipartFile configFile) throws IOException {
        String script = scriptPath + "/train_" + record.getModelType().toLowerCase() + ".py";
        
        // 保存配置文件
        String configPath = null;
        if (configFile != null) {
            Path configDir = Paths.get(scriptPath);
            Files.createDirectories(configDir);
            configPath = configDir.resolve("config_" + record.getId() + ".yaml");
            Files.copy(configFile.getInputStream(), configPath);
        }
        
        ProcessBuilder pb = new ProcessBuilder(
                pythonPath,
                script,
                "--model", record.getModelType(),
                "--train_dataset", record.getTrainDataset(),
                "--test_dataset", record.getTestDataset(),
                "--epochs", String.valueOf(record.getEpochs()),
                "--batch_size", String.valueOf(record.getBatchSize()),
                "--learning_rate", String.valueOf(record.getLearningRate()),
                "--output", record.getModelPath()
        );
        
        if (configPath != null) {
            pb.command().add("--config");
            pb.command().add(configPath);
        }
        
        pb.redirectErrorStream(true);
        Process process = pb.start();
        
        // 读取输出
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info(line);
            }
        }
        
        try {
            process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // 解析训练结果
        return parseTrainingResult(output.toString(), record);
    }
    
    /**
     * 解析训练结果
     */
    private TrainingResult parseTrainingResult(String output, TrainingRecord record) {
        TrainingResult result = new TrainingResult();
        
        // 简单解析输出中的指标
        String[] lines = output.split("\n");
        for (String line : lines) {
            if (line.contains("train_accuracy")) {
                result.trainAccuracy = extractDouble(line);
            } else if (line.contains("val_accuracy")) {
                result.valAccuracy = extractDouble(line);
            } else if (line.contains("test_accuracy")) {
                result.testAccuracy = extractDouble(line);
            } else if (line.contains("mean_iou")) {
                result.meanIou = extractDouble(line);
            } else if (line.contains("precision")) {
                result.precision = extractDouble(line);
            } else if (line.contains("recall")) {
                result.recall = extractDouble(line);
            } else if (line.contains("f1_score")) {
                result.f1Score = extractDouble(line);
            }
        }
        
        result.logPath = record.getModelPath() + "/training.log";
        return result;
    }
    
    private Double extractDouble(String line) {
        try {
            String[] parts = line.split(":");
            if (parts.length > 1) {
                return Double.parseDouble(parts[1].trim());
            }
        } catch (Exception ignored) {}
        return 0.0;
    }
    
    /**
     * 获取训练记录列表
     */
    public List<TrainingRecord> getTrainingRecords(int page, int size) {
        return trainingRecordRepository.findAll(
                org.springframework.data.domain.PageRequest.of(page, size,
                        org.springframework.data.domain.Sort.by("createdAt").descending()))
                .getContent();
    }
    
    /**
     * 获取指定模型的训练历史
     */
    public List<TrainingRecord> getTrainingHistory(String modelType) {
        return trainingRecordRepository.findTop10ByModelTypeOrderByCreatedAtDesc(modelType);
    }
    
    /**
     * 获取训练记录详情
     */
    public TrainingRecord getTrainingRecord(Long id) {
        return trainingRecordRepository.findById(id).orElse(null);
    }
    
    /**
     * 获取所有可用模型
     */
    public List<String> getAvailableModels() {
        return List.of("UNET", "MASK_RCNN", "YOLO");
    }
    
    /**
     * 内部类：训练结果
     */
    private static class TrainingResult {
        Double trainAccuracy = 0.0;
        Double valAccuracy = 0.0;
        Double testAccuracy = 0.0;
        Double meanIou = 0.0;
        Double precision = 0.0;
        Double recall = 0.0;
        Double f1Score = 0.0;
        String logPath;
    }
}
