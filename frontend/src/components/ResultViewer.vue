<template>
  <div class="result-viewer">
    <el-card class="result-card">
      <template #header>
        <div class="card-header">
          <span>检测结果</span>
          <el-tag :type="resultTagType" size="small">
            {{ resultTagText }}
          </el-tag>
        </div>
      </template>

      <!-- 统计概览 -->
      <div class="stats-overview">
        <el-row :gutter="20">
          <el-col :span="8">
            <div class="stat-item">
              <div class="stat-value">{{ result.nematode_count || 0 }}</div>
              <div class="stat-label">检测数量</div>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="stat-item">
              <div class="stat-value">{{ (result.confidence * 100).toFixed(1) }}%</div>
              <div class="stat-label">置信度</div>
            </div>
          </el-col>
          <el-col :span="8">
            <div class="stat-item">
              <div class="stat-value">{{ result.processing_time?.toFixed(3) || '-' }}s</div>
              <div class="stat-label">处理时间</div>
            </div>
          </el-col>
        </el-row>
      </div>

      <!-- 结果图表 -->
      <div v-if="result.results && result.results.length > 0" class="results-chart">
        <h4>检测详情</h4>
        <el-table :data="result.results" stripe style="width: 100%">
          <el-table-column prop="index" label="#" width="60">
            <template #default="{ $index }">
              {{ $index + 1 }}
            </template>
          </el-table-column>
          <el-table-column prop="bbox" label="位置 [x1,y1,x2,y2]" min-width="160">
            <template #default="{ row }">
              {{ row.bbox?.join(', ') || '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="confidence" label="置信度" width="100">
            <template #default="{ row }">
              <el-tag type="success" size="small">
                {{ (row.confidence * 100).toFixed(1) }}%
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="area" label="面积(px)" width="100" />
        </el-table>
      </div>

      <!-- 可视化结果 -->
      <div v-if="resultImage" class="visualization">
        <h4>可视化结果</h4>
        <el-image
          :src="resultImage"
          :preview-src-list="[resultImage]"
          fit="contain"
          class="result-image"
        />
      </div>

      <!-- 操作按钮 -->
      <div class="actions">
        <el-button type="primary" @click="handleSave">
          <i class="el-icon-download"></i> 保存结果
        </el-button>
        <el-button @click="handleReset">
          <i class="el-icon-refresh"></i> 重新检测
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script>
import { defineComponent, computed } from 'vue'

export default defineComponent({
  name: 'ResultViewer',
  props: {
    result: {
      type: Object,
      default: () => ({})
    },
    resultImage: {
      type: String,
      default: ''
    },
    modelType: {
      type: String,
      default: 'UNET'
    }
  },
  emits: ['save', 'reset'],
  setup(props, { emit }) {
    const resultTagType = computed(() => {
      const count = props.result.nematode_count || 0
      if (count === 0) return 'info'
      if (count <= 3) return 'success'
      if (count <= 10) return 'warning'
      return 'danger'
    })

    const resultTagText = computed(() => {
      const count = props.result.nematode_count || 0
      if (count === 0) return '未检测到'
      return `检测到 ${count} 处`
    })

    const handleSave = () => emit('save', props.result)
    const handleReset = () => emit('reset')

    return {
      resultTagType,
      resultTagText,
      handleSave,
      handleReset
    }
  }
})
</script>

<style scoped>
.result-viewer {
  width: 100%;
}

.result-card {
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stats-overview {
  padding: 20px 0;
  border-bottom: 1px solid #ebeef5;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #409eff;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 8px;
}

.results-chart,
.visualization {
  margin-top: 20px;
}

.results-chart h4,
.visualization h4 {
  margin-bottom: 15px;
  color: #303133;
}

.result-image {
  width: 100%;
  max-height: 500px;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.actions {
  margin-top: 20px;
  display: flex;
  gap: 10px;
  justify-content: center;
}
</style>
