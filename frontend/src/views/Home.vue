<template>
  <div class="home-container">
    <el-row :gutter="20">
      <!-- 欢迎卡片 -->
      <el-col :span="24">
        <el-card class="welcome-card">
          <template #header>
            <div class="card-header">
              <span>欢迎使用松材线虫病检测系统</span>
            </div>
          </template>
          <div class="welcome-content">
            <p>基于深度学习的CT图像松材线虫病自动检测系统</p>
            <p class="description">
              本系统提供三种先进的深度学习模型进行松材线虫检测：
            </p>
            <ul>
              <li><strong>改进U-Net</strong> - 结合CBAM注意力机制，分割精度高</li>
              <li><strong>Mask R-CNN</strong> - 实例分割，可区分不同目标</li>
              <li><strong>YOLO</strong> - 实时目标检测，速度快</li>
            </ul>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="stats-row">
      <!-- 统计数据 -->
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#409EFF"><Document /></el-icon>
            <div class="stat-info">
              <span class="stat-value">{{ stats.totalImages }}</span>
              <span class="stat-label">已检测图像</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#67C23A"><SuccessFilled /></el-icon>
            <div class="stat-info">
              <span class="stat-value">{{ stats.detectedCount }}</span>
              <span class="stat-label">检测到线虫</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#E6A23C"><Cpu /></el-icon>
            <div class="stat-info">
              <span class="stat-value">{{ stats.modelCount }}</span>
              <span class="stat-label">可用模型</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#F56C6C"><Timer /></el-icon>
            <div class="stat-info">
              <span class="stat-value">{{ stats.avgTime }}s</span>
              <span class="stat-label">平均检测时间</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20">
      <!-- 快速开始 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>快速开始</span>
            </div>
          </template>
          <div class="quick-start">
            <el-steps direction="vertical" :space="60">
              <el-step title="上传CT图像" description="支持PNG、JPG格式" />
              <el-step title="选择检测模型" description="U-Net、Mask R-CNN、YOLO" />
              <el-step title="执行检测" description="自动检测松材线虫" />
              <el-step title="查看结果" description="可视化检测结果" />
            </el-steps>
            <el-button type="primary" class="start-btn" @click="$router.push('/detect')">
              开始检测
            </el-button>
          </div>
        </el-card>
      </el-col>

      <!-- 最新检测结果 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>最新检测记录</span>
              <el-button type="text" @click="$router.push('/history')">查看更多</el-button>
            </div>
          </template>
          <el-table :data="recentRecords" style="width: 100%">
            <el-table-column prop="id" label="ID" width="60" />
            <el-table-column prop="modelType" label="模型" width="100" />
            <el-table-column prop="nematodeCount" label="线虫数" width="80" />
            <el-table-column prop="confidence" label="置信度" width="80">
              <template #default="{ row }">
                {{ (row.confidence * 100).toFixed(1) }}%
              </template>
            </el-table-column>
            <el-table-column prop="createdAt" label="时间" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Document, SuccessFilled, Cpu, Timer } from '@element-plus/icons-vue'
import { detectionAPI } from '@/api'

const stats = ref({
  totalImages: 156,
  detectedCount: 89,
  modelCount: 3,
  avgTime: '1.2'
})

const recentRecords = ref([
  { id: 1, modelType: 'UNET', nematodeCount: 3, confidence: 0.92, createdAt: '2024-01-15 10:30' },
  { id: 2, modelType: 'YOLO', nematodeCount: 5, confidence: 0.88, createdAt: '2024-01-15 10:25' },
  { id: 3, modelType: 'MASK_RCNN', nematodeCount: 2, confidence: 0.95, createdAt: '2024-01-15 10:20' }
])

onMounted(() => {
  fetchRecentRecords()
})

const fetchRecentRecords = async () => {
  try {
    const response = await detectionAPI.getRecords(0, 5)
    if (response.records) {
      recentRecords.value = response.records.map(r => ({
        ...r,
        createdAt: new Date(r.createdAt).toLocaleString()
      }))
    }
  } catch (error) {
    console.error('获取检测记录失败:', error)
  }
}
</script>

<style lang="scss" scoped>
.home-container {
  .welcome-card {
    margin-bottom: 20px;
    
    .welcome-content {
      p {
        margin: 10px 0;
      }
      
      .description {
        color: #666;
      }
      
      ul {
        margin-top: 15px;
        padding-left: 20px;
        
        li {
          margin: 8px 0;
        }
      }
    }
  }
  
  .stats-row {
    margin-bottom: 20px;
    
    .stat-card {
      .stat-content {
        display: flex;
        align-items: center;
        gap: 15px;
        
        .stat-icon {
          font-size: 40px;
        }
        
        .stat-info {
          display: flex;
          flex-direction: column;
          
          .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #333;
          }
          
          .stat-label {
            font-size: 14px;
            color: #999;
          }
        }
      }
    }
  }
  
  .quick-start {
    .start-btn {
      margin-top: 20px;
      width: 100%;
    }
  }
}
</style>
