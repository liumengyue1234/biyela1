<template>
  <div class="detect-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>CT图像松材线虫检测</span>
        </div>
      </template>
      
      <el-row :gutter="20">
        <!-- 左侧：上传和参数设置 -->
        <el-col :span="8">
          <div class="upload-section">
            <el-upload
              class="upload-area"
              drag
              :auto-upload="false"
              :limit="1"
              :on-change="handleFileChange"
              :file-list="fileList"
              accept=".png,.jpg,.jpeg"
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">
                拖拽图像到此处或<em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  支持PNG、JPG格式，文件大小不超过100MB
                </div>
              </template>
            </el-upload>
            
            <div class="preview-image" v-if="previewUrl">
              <img :src="previewUrl" alt="预览图像" />
            </div>
            
            <!-- 检测参数 -->
            <div class="detection-params">
              <h4>检测参数</h4>
              
              <el-form label-width="80px">
                <el-form-item label="检测模型">
                  <el-select v-model="params.modelType" placeholder="选择模型">
                    <el-option label="改进U-Net" value="UNET" />
                    <el-option label="Mask R-CNN" value="MASK_RCNN" />
                    <el-option label="YOLOv8" value="YOLOv8" />
                    <el-option label="YOLO11" value="YOLO11" />
                    <el-option label="YOLOv10" value="YOLOv10" />
                  </el-select>
                </el-form-item>
                
                <el-form-item label="置信度阈值">
                  <el-slider 
                    v-model="params.threshold" 
                    :min="0.1" 
                    :max="1" 
                    :step="0.05"
                    show-input
                  />
                </el-form-item>
              </el-form>
              
              <el-button 
                type="primary" 
                :loading="detecting"
                :disabled="!selectedFile"
                @click="startDetection"
                class="detect-btn"
              >
                {{ detecting ? '检测中...' : '开始检测' }}
              </el-button>
            </div>
          </div>
        </el-col>
        
        <!-- 右侧：结果显示 -->
        <el-col :span="16">
          <div class="result-section">
            <el-tabs v-model="activeTab">
              <el-tab-pane label="检测结果" name="result">
                <div class="result-display" v-if="detectionResult">
                  <el-row :gutter="20">
                    <el-col :span="12">
                      <div class="result-image">
                        <h4>原图</h4>
                        <img :src="previewUrl" alt="原图" />
                      </div>
                    </el-col>
                    <el-col :span="12">
                      <div class="result-image">
                        <h4>检测结果</h4>
                        <img v-if="resultImageUrl" :src="resultImageUrl" alt="检测结果" />
                        <div v-else class="result-placeholder">
                          检测结果将显示在这里
                        </div>
                      </div>
                    </el-col>
                  </el-row>
                  
                  <el-divider />
                  
                  <!-- 统计信息 -->
                  <el-descriptions :column="3" border>
                    <el-descriptions-item label="检测到的线虫数量">
                      <el-tag type="danger">{{ detectionResult.nematodeCount }}</el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="平均置信度">
                      <el-tag type="success">{{ (detectionResult.confidence * 100).toFixed(2) }}%</el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="处理时间">
                      <el-tag>{{ detectionResult.processingTime }}ms</el-tag>
                    </el-descriptions-item>
                  </el-descriptions>
                  
                  <!-- 详细检测结果 -->
                  <el-table 
                    :data="detectionResult.results" 
                    style="width: 100%; margin-top: 20px"
                    v-if="detectionResult.results && detectionResult.results.length > 0"
                  >
                    <el-table-column prop="bbox" label="边界框" width="180">
                      <template #default="{ row }">
                        [{{ row.bbox.join(', ') }}]
                      </template>
                    </el-table-column>
                    <el-table-column prop="confidence" label="置信度" width="120">
                      <template #default="{ row }">
                        {{ (row.confidence * 100).toFixed(2) }}%
                      </template>
                    </el-table-column>
                    <el-table-column prop="area" label="区域面积" />
                  </el-table>
                </div>
                
                <div v-else class="no-result">
                  <el-empty description="请上传图像并点击检测按钮" />
                </div>
              </el-tab-pane>
              
              <el-tab-pane label="检测历史" name="history">
                <el-table :data="historyList" style="width: 100%">
                  <el-table-column prop="id" label="ID" width="60" />
                  <el-table-column prop="modelType" label="模型" width="100" />
                  <el-table-column prop="nematodeCount" label="线虫数" width="80" />
                  <el-table-column prop="confidence" label="置信度" width="100">
                    <template #default="{ row }">
                      {{ (row.confidence * 100).toFixed(1) }}%
                    </template>
                  </el-table-column>
                  <el-table-column prop="status" label="状态" width="100">
                    <template #default="{ row }">
                      <el-tag :type="row.status === 'COMPLETED' ? 'success' : 'info'">
                        {{ row.status }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column prop="createdAt" label="时间" />
                </el-table>
              </el-tab-pane>
            </el-tabs>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { detectionAPI } from '@/api'

const fileList = ref([])
const selectedFile = ref(null)
const previewUrl = ref('')
const detecting = ref(false)
const activeTab = ref('result')
const detectionResult = ref(null)
const resultImageUrl = ref('')
const historyList = ref([])

const params = reactive({
  modelType: 'UNET',
  threshold: 0.5
})

const handleFileChange = (file) => {
  selectedFile.value = file.raw
  previewUrl.value = URL.createObjectURL(file.raw)
  detectionResult.value = null
  resultImageUrl.value = ''
}

const startDetection = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择图像')
    return
  }
  
  detecting.value = true
  
  try {
    const response = await detectionAPI.uploadAndDetect(
      selectedFile.value,
      params.modelType,
      params.threshold
    )
    
    detectionResult.value = response
    if (response.resultImageUrl) {
      resultImageUrl.value = response.resultImageUrl
    }
    
    ElMessage.success('检测完成！')
  } catch (error) {
    console.error('检测失败:', error)
    ElMessage.error('检测失败，请重试')
  } finally {
    detecting.value = false
  }
}

onMounted(() => {
  fetchHistory()
})

const fetchHistory = async () => {
  try {
    const response = await detectionAPI.getRecords(0, 10)
    if (response.records) {
      historyList.value = response.records.map(r => ({
        ...r,
        createdAt: new Date(r.createdAt).toLocaleString()
      }))
    }
  } catch (error) {
    console.error('获取历史记录失败:', error)
  }
}
</script>

<style lang="scss" scoped>
.detect-container {
  .upload-section {
    .upload-area {
      margin-bottom: 20px;
    }
    
    .preview-image {
      margin-bottom: 20px;
      
      img {
        width: 100%;
        border-radius: 8px;
      }
    }
    
    .detection-params {
      background: #f5f7fa;
      padding: 15px;
      border-radius: 8px;
      
      h4 {
        margin-bottom: 15px;
      }
      
      .detect-btn {
        width: 100%;
        margin-top: 15px;
      }
    }
  }
  
  .result-section {
    .result-display {
      .result-image {
        background: #f5f7fa;
        padding: 10px;
        border-radius: 8px;
        
        h4 {
          margin-bottom: 10px;
        }
        
        img {
          width: 100%;
          border-radius: 4px;
        }
        
        .result-placeholder {
          height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #fff;
          border: 1px dashed #ddd;
          border-radius: 4px;
          color: #999;
        }
      }
    }
    
    .no-result {
      height: 400px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
  }
}
</style>
