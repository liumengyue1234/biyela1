package com.bio.detect.repository;

import com.bio.detect.model.DetectionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import java.util.List;

/**
 * 检测记录Repository
 */
@Repository
public interface DetectionRecordRepository extends JpaRepository<DetectionRecord, Long> {
    
    /**
     * 根据模型类型查询检测记录
     */
    List<DetectionRecord> findByModelType(String modelType);
    
    /**
     * 根据状态查询检测记录
     */
    List<DetectionRecord> findByStatus(String status);
    
    /**
     * 查询最近的检测记录
     */
    @Query(value = "SELECT * FROM detection_records ORDER BY created_at DESC LIMIT ?1", nativeQuery = true)
    List<DetectionRecord> findRecentRecords(int limit);
    
    /**
     * 统计各模型的检测数量
     */
    @Query(value = "SELECT model_type, COUNT(*) FROM detection_records GROUP BY model_type", nativeQuery = true)
    List<Object[]> countByModelType();
}
