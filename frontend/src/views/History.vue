<template>
  <div class="history-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>检测记录</span>
          <el-button type="primary" @click="fetchRecords">
            刷新
          </el-button>
        </div>
      </template>
      
      <el-table :data="records" style="width: 100%" v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column label="图像" width="120">
          <template #default="{ row }">
            <el-image 
              v-if="row.imagePath" 
              :src="getImageUrl(row.imagePath)" 
              style="width: 80px; height: 80px"
              fit="cover"
            />
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="modelType" label="模型" width="120">
          <template #default="{ row }">
            <el-tag>{{ row.modelType }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="nematodeCount" label="线虫数量" width="100">
          <template #default="{ row }">
            <el-tag type="danger">{{ row.nematodeCount }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            {{ (row.confidence * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="processingTime" label="处理时间" width="100">
          <template #default="{ row }">
            {{ row.processingTime }}ms
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="检测时间" />
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button size="small" @click="viewDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="fetchRecords"
        @current-change="fetchRecords"
        style="margin-top: 20px; text-align: right"
      />
    </el-card>
    
    <!-- 详情对话框 -->
    <el-dialog v-model="showDetail" title="检测详情" width="800px">
      <div v-if="selectedRecord" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ID">{{ selectedRecord.id }}</el-descriptions-item>
          <el-descriptions-item label="模型">{{ selectedRecord.modelType }}</el-descriptions-item>
          <el-descriptions-item label="线虫数量">{{ selectedRecord.nematodeCount }}</el-descriptions-item>
          <el-descriptions-item label="置信度">{{ (selectedRecord.confidence * 100).toFixed(2) }}%</el-descriptions-item>
          <el-descriptions-item label="状态">{{ selectedRecord.status }}</el-descriptions-item>
          <el-descriptions-item label="处理时间">{{ selectedRecord.processingTime }}ms</el-descriptions-item>
        </el-descriptions>
        
        <el-divider />
        
        <div class="image-comparison" v-if="selectedRecord.imagePath">
          <el-row :gutter="20">
            <el-col :span="12">
              <h4>原图</h4>
              <img :src="getImageUrl(selectedRecord.imagePath)" style="width: 100%" />
            </el-col>
            <el-col :span="12" v-if="selectedRecord.resultImagePath">
              <h4>检测结果</h4>
              <img :src="getImageUrl(selectedRecord.resultImagePath)" style="width: 100%" />
            </el-col>
          </el-row>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { detectionAPI } from '@/api'

const records = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)
const showDetail = ref(false)
const selectedRecord = ref(null)

const fetchRecords = async () => {
  loading.value = true
  try {
    const response = await detectionAPI.getRecords(currentPage.value - 1, pageSize.value)
    if (response.records) {
      records.value = response.records.map(r => ({
        ...r,
        createdAt: new Date(r.createdAt).toLocaleString()
      }))
      total.value = response.total
    }
  } catch (error) {
    console.error('获取记录失败:', error)
  } finally {
    loading.value = false
  }
}

const getStatusType = (status) => {
  const types = {
    'COMPLETED': 'success',
    'PROCESSING': 'warning',
    'FAILED': 'danger',
    'PENDING': 'info'
  }
  return types[status] || 'info'
}

const getImageUrl = (path) => {
  if (!path) return ''
  // 如果是本地路径，转换为URL
  return path.startsWith('http') ? path : `http://localhost:8080/api/${path}`
}

const viewDetail = (row) => {
  selectedRecord.value = row
  showDetail.value = true
}

onMounted(() => {
  fetchRecords()
})
</script>

<style lang="scss" scoped>
.history-container {
  .image-comparison {
    margin-top: 20px;
    
    h4 {
      margin-bottom: 10px;
    }
  }
}
</style>
