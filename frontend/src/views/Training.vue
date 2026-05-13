<template>
  <div class="training-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>模型训练</span>
          <el-button type="primary" @click="showTrainDialog = true">
            开始训练
          </el-button>
        </div>
      </template>
      
      <el-tabs v-model="activeTab">
        <el-tab-pane label="训练管理" name="manage">
          <!-- 模型选择 -->
          <el-row :gutter="20" class="model-cards">
            <el-col :span="8" v-for="model in models" :key="model.type">
              <el-card class="model-card" shadow="hover">
                <template #header>
                  <div class="model-header">
                    <span>{{ model.name }}</span>
                    <el-tag :type="model.status === 'available' ? 'success' : 'info'">
                      {{ model.status }}
                    </el-tag>
                  </div>
                </template>
                <div class="model-info">
                  <p><strong>类型：</strong>{{ model.type }}</p>
                  <p><strong>参数量：</strong>{{ model.params }}</p>
                  <p><strong>最新准确率：</strong>{{ model.accuracy }}</p>
                </div>
                <el-button type="primary" size="small" @click="selectModel(model)">
                  选择训练
                </el-button>
              </el-card>
            </el-col>
          </el-row>
          
          <!-- 训练记录 -->
          <el-divider />
          <h3>训练记录</h3>
          <el-table :data="trainingRecords" style="width: 100%">
            <el-table-column prop="id" label="ID" width="60" />
            <el-table-column prop="modelType" label="模型" width="120" />
            <el-table-column prop="epochs" label="轮数" width="80" />
            <el-table-column prop="status" label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">
                  {{ row.status }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="testAccuracy" label="测试准确率" width="120">
              <template #default="{ row }">
                {{ row.testAccuracy ? (row.testAccuracy * 100).toFixed(2) + '%' : '-' }}
              </template>
            </el-table-column>
            <el-table-column prop="createdAt" label="训练时间" />
          </el-table>
        </el-tab-pane>
        
        <el-tab-pane label="训练监控" name="monitor">
          <el-row :gutter="20">
            <el-col :span="12">
              <div id="lossChart" style="height: 300px"></div>
            </el-col>
            <el-col :span="12">
              <div id="accuracyChart" style="height: 300px"></div>
            </el-col>
          </el-row>
        </el-tab-pane>
      </el-tabs>
    </el-card>
    
    <!-- 训练配置对话框 -->
    <el-dialog v-model="showTrainDialog" title="训练配置" width="600px">
      <el-form :model="trainConfig" label-width="100px">
        <el-form-item label="模型类型">
          <el-select v-model="trainConfig.modelType">
            <el-option label="改进U-Net" value="UNET" />
            <el-option label="Mask R-CNN" value="MASK_RCNN" />
            <el-option label="YOLOv8" value="YOLOv8" />
            <el-option label="YOLO11" value="YOLO11" />
            <el-option label="YOLOv10" value="YOLOv10" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="训练数据集">
          <el-select v-model="trainConfig.trainDataset">
            <el-option label="First数据集" value="first" />
            <el-option label="Third数据集" value="third" />
            <el-option label="合并数据集" value="combined" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="训练轮数">
          <el-input-number v-model="trainConfig.epochs" :min="1" :max="500" />
        </el-form-item>
        
        <el-form-item label="批次大小">
          <el-input-number v-model="trainConfig.batchSize" :min="1" :max="64" />
        </el-form-item>
        
        <el-form-item label="学习率">
          <el-input-number v-model="trainConfig.learningRate" :min="0.0001" :max="0.1" :step="0.001" />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showTrainDialog = false">取消</el-button>
        <el-button type="primary" @click="startTraining">开始训练</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { trainingAPI } from '@/api'
import * as echarts from 'echarts'

const activeTab = ref('manage')
const showTrainDialog = ref(false)

const models = ref([
  { type: 'UNET', name: '改进U-Net', status: 'available', params: '31.2M', accuracy: '89.5%' },
  { type: 'MASK_RCNN', name: 'Mask R-CNN', status: 'available', params: '63.2M', accuracy: '87.3%' },
  { type: 'YOLOv8', name: 'YOLOv8', status: 'available', params: '11.2M', accuracy: '85.8%' },
  { type: 'YOLO11', name: 'YOLO11', status: 'available', params: '25.9M', accuracy: '86.5%' },
  { type: 'YOLOv10', name: 'YOLOv10', status: 'available', params: '7.2M', accuracy: '84.2%' }
])

const trainingRecords = ref([])

const trainConfig = reactive({
  modelType: 'UNET',
  trainDataset: 'combined',
  epochs: 50,
  batchSize: 8,
  learningRate: 0.001
})

const selectModel = (model) => {
  trainConfig.modelType = model.type
  showTrainDialog.value = true
}

const startTraining = async () => {
  try {
    const response = await trainingAPI.startTraining(trainConfig)
    ElMessage.success('训练已启动！')
    showTrainDialog.value = false
    fetchTrainingRecords()
  } catch (error) {
    console.error('启动训练失败:', error)
    ElMessage.error('启动训练失败，请重试')
  }
}

const getStatusType = (status) => {
  const types = {
    'COMPLETED': 'success',
    'TRAINING': 'warning',
    'FAILED': 'danger',
    'PENDING': 'info'
  }
  return types[status] || 'info'
}

onMounted(() => {
  fetchTrainingRecords()
  initCharts()
})

const fetchTrainingRecords = async () => {
  try {
    const response = await trainingAPI.getRecords(0, 20)
    if (response.records) {
      trainingRecords.value = response.records.map(r => ({
        ...r,
        createdAt: new Date(r.createdAt).toLocaleString()
      }))
    }
  } catch (error) {
    console.error('获取训练记录失败:', error)
  }
}

const initCharts = () => {
  nextTick(() => {
    const lossChart = echarts.init(document.getElementById('lossChart'))
    lossChart.setOption({
      title: { text: '训练损失' },
      tooltip: {},
      xAxis: { type: 'category', data: ['1', '2', '3', '4', '5'] },
      yAxis: { type: 'value' },
      series: [{ data: [0.5, 0.35, 0.28, 0.22, 0.18], type: 'line', smooth: true }]
    })
    
    const accuracyChart = echarts.init(document.getElementById('accuracyChart'))
    accuracyChart.setOption({
      title: { text: '验证准确率' },
      tooltip: {},
      xAxis: { type: 'category', data: ['1', '2', '3', '4', '5'] },
      yAxis: { type: 'value', max: 1 },
      series: [{ data: [0.65, 0.75, 0.82, 0.86, 0.89], type: 'line', smooth: true }]
    })
  })
}
</script>

<style lang="scss" scoped>
.training-container {
  .model-cards {
    margin-bottom: 20px;
    
    .model-card {
      .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      
      .model-info {
        margin-bottom: 15px;
        
        p {
          margin: 5px 0;
        }
      }
    }
  }
}
</style>
