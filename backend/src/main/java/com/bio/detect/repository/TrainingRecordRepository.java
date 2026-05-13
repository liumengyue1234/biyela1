package com.bio.detect.repository;

import com.bio.detect.model.TrainingRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

/**
 * 训练记录Repository
 */
@Repository
public interface TrainingRecordRepository extends JpaRepository<TrainingRecord, Long> {
    
    /**
     * 根据模型类型查询训练记录
     */
    List<TrainingRecord> findByModelType(String modelType);
    
    /**
     * 根据状态查询训练记录
     */
    List<TrainingRecord> findByStatus(String status);
    
    /**
     * 查询最新训练记录
     */
    List<TrainingRecord> findTop10ByModelTypeOrderByCreatedAtDesc(String modelType);
}
