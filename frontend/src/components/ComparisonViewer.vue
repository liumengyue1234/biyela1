<template>
  <div class="comparison-viewer">
    <el-card class="comparison-card">
      <template #header>
        <div class="card-header">
          <span>模型对比分析</span>
          <el-button type="primary" size="small" @click="handleExport">
            <i class="el-icon-download"></i> 导出报告
          </el-button>
        </div>
      </template>

      <!-- 图片网格 -->
      <div class="image-grid">
        <div v-for="(item, index) in comparisonData" :key="index" class="image-item">
          <el-image
            :src="item.image"
            :preview-src-list="[item.image]"
            fit="contain"
            class="preview-image"
          />
          <div class="image-info">
            <span class="image-name">{{ item.name }}</span>
            <span class="image-count">数量: {{ item.count }}</span>
            <span class="image-conf">置信度: {{ (item.confidence * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>

      <!-- 统计表格 -->
      <div class="stats-table">
        <el-table :data="statsData" stripe border>
          <el-table-column prop="model" label="模型" width="120" />
          <el-table-column prop="count" label="检测数量" width="100" align="center" />
          <el-table-column prop="confidence" label="平均置信度" width="120" align="center">
            <template #default="{ row }">
              <el-progress
                :percentage="(row.confidence * 100)"
                :color="getProgressColor(row.confidence)"
                :format="() => (row.confidence * 100).toFixed(1) + '%'"
              />
            </template>
          </el-table-column>
          <el-table-column prop="time" label="处理时间" width="100" align="center" />
          <el-table-column prop="accuracy" label="测试准确率" width="120" align="center">
            <template #default="{ row }">
              <el-tag type="success">{{ row.accuracy }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="miou" label="Mean IoU" width="100" align="center" />
          <el-table-column prop="f1" label="F1 Score" width="100" align="center" />
        </el-table>
      </div>

      <!-- 雷达图 -->
      <div class="radar-chart">
        <div ref="radarRef" style="width: 100%; height: 400px;"></div>
      </div>
    </el-card>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted, computed } from 'vue'
import * as echarts from 'echarts'

export default defineComponent({
  name: 'ComparisonViewer',
  props: {
    comparisonData: {
      type: Array,
      default: () => []
    }
  },
  emits: ['export'],
  setup(props, { emit }) {
    const radarRef = ref(null)

    const statsData = ref([
      {
        model: '改进 U-Net',
        count: 0,
        confidence: 0.878,
        time: '-',
        accuracy: '89.5%',
        miou: '82.3%',
        f1: '0.878'
      },
      {
        model: 'Mask R-CNN',
        count: 0,
        confidence: 0.852,
        time: '-',
        accuracy: '87.3%',
        miou: '79.8%',
        f1: '0.852'
      },
      {
        model: 'YOLO',
        count: 0,
        confidence: 0.831,
        time: '-',
        accuracy: '85.8%',
        miou: '76.5%',
        f1: '0.831'
      }
    ])

    const getProgressColor = (value) => {
      if (value >= 0.85) return '#67c23a'
      if (value >= 0.70) return '#e6a23c'
      return '#f56c6c'
    }

    const handleExport = () => emit('export', statsData.value)

    const initRadarChart = () => {
      if (!radarRef.value) return

      const chart = echarts.init(radarRef.value)

      const option = {
        title: {
          text: '模型性能雷达图对比',
          left: 'center'
        },
        tooltip: {
          trigger: 'item'
        },
        legend: {
          data: ['U-Net', 'Mask R-CNN', 'YOLO'],
          bottom: 10
        },
        radar: {
          indicator: [
            { name: '准确率', max: 100 },
            { name: 'Mean IoU', max: 100 },
            { name: 'F1 Score', max: 1 },
            { name: '速度', max: 1 },
            { name: '稳定性', max: 1 }
          ],
          radius: '65%'
        },
        series: [
          {
            type: 'radar',
            data: [
              {
                value: [89.5, 82.3, 0.878, 0.6, 0.85],
                name: 'U-Net',
                itemStyle: { color: '#409eff' }
              },
              {
                value: [87.3, 79.8, 0.852, 0.4, 0.82],
                name: 'Mask R-CNN',
                itemStyle: { color: '#67c23a' }
              },
              {
                value: [85.8, 76.5, 0.831, 0.9, 0.78],
                name: 'YOLO',
                itemStyle: { color: '#e6a23c' }
              }
            ]
          }
        ]
      }

      chart.setOption(option)
    }

    onMounted(() => {
      initRadarChart()
    })

    return {
      radarRef,
      statsData,
      getProgressColor,
      handleExport
    }
  }
})
</script>

<style scoped>
.comparison-viewer {
  width: 100%;
}

.comparison-card {
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 20px;
}

.image-item {
  text-align: center;
}

.preview-image {
  width: 100%;
  height: 200px;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.image-info {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 10px;
}

.image-name {
  font-weight: bold;
  color: #303133;
}

.image-count,
.image-conf {
  font-size: 13px;
  color: #606266;
}

.stats-table {
  margin-bottom: 20px;
}

.radar-chart {
  margin-top: 20px;
}
</style>
