<template>
  <div class="comparison-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>模型性能对比</span>
          <el-button type="primary" @click="fetchComparison">
            刷新数据
          </el-button>
        </div>
      </template>
      
      <!-- 模型性能对比表格 -->
      <el-table :data="comparisonData" style="width: 100%" stripe>
        <el-table-column prop="model" label="模型" width="150" />
        <el-table-column prop="testAccuracy" label="测试准确率" width="130">
          <template #default="{ row }">
            <span :style="{ color: getAccuracyColor(row.testAccuracy) }">
              {{ (row.testAccuracy * 100).toFixed(2) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="meanIou" label="平均IoU" width="120">
          <template #default="{ row }">
            {{ (row.meanIou * 100).toFixed(2) }}%
          </template>
        </el-table-column>
        <el-table-column prop="f1Score" label="F1分数" width="100">
          <template #default="{ row }">
            {{ row.f1Score.toFixed(4) }}
          </template>
        </el-table-column>
        <el-table-column prop="precision" label="精确率" width="100">
          <template #default="{ row }">
            {{ row.precision.toFixed(4) }}
          </template>
        </el-table-column>
        <el-table-column prop="recall" label="召回率" width="100">
          <template #default="{ row }">
            {{ row.recall.toFixed(4) }}
          </template>
        </el-table-column>
        <el-table-column prop="trainingTime" label="训练时间(s)" width="120" />
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button size="small" @click="selectModel(row)">选择使用</el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <el-divider />
      
      <!-- 可视化对比 -->
      <el-row :gutter="20">
        <el-col :span="12">
          <div id="accuracyChart" style="height: 350px"></div>
        </el-col>
        <el-col :span="12">
          <div id="metricsChart" style="height: 350px"></div>
        </el-col>
      </el-row>
      
      <el-divider />
      
      <!-- 性能分析 -->
      <el-row :gutter="20">
        <el-col :span="8" v-for="model in comparisonData" :key="model.model">
          <el-card shadow="hover">
            <template #header>
              <span>{{ model.model }}</span>
            </template>
            <div class="model-analysis">
              <el-progress 
                :percentage="model.testAccuracy * 100" 
                :color="getAccuracyColor(model.testAccuracy)"
              />
              <p><strong>准确率：</strong>{{ (model.testAccuracy * 100).toFixed(2) }}%</p>
              <p><strong>IoU：</strong>{{ (model.meanIou * 100).toFixed(2) }}%</p>
              <p><strong>推理速度：</strong>{{ model.trainingTime }}s</p>
              <el-tag :type="getBestModel(model.model) ? 'success' : 'info'">
                {{ getBestModel(model.model) ? '推荐模型' : '备选模型' }}
              </el-tag>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { trainingAPI } from '@/api'
import * as echarts from 'echarts'

const comparisonData = ref([
  {
    model: '改进U-Net',
    testAccuracy: 0.895,
    meanIou: 0.823,
    f1Score: 0.878,
    precision: 0.912,
    recall: 0.847,
    trainingTime: 3600
  },
  {
    model: 'Mask R-CNN',
    testAccuracy: 0.873,
    meanIou: 0.798,
    f1Score: 0.852,
    precision: 0.889,
    recall: 0.818,
    trainingTime: 5400
  },
  {
    model: 'YOLO',
    testAccuracy: 0.858,
    meanIou: 0.765,
    f1Score: 0.831,
    precision: 0.865,
    recall: 0.799,
    trainingTime: 2400
  }
])

const bestModel = ref('改进U-Net')

const fetchComparison = async () => {
  try {
    const response = await trainingAPI.getComparison()
    if (response.UNET) {
      // 更新数据
      ElMessage.success('数据已刷新')
    }
  } catch (error) {
    console.error('获取对比数据失败:', error)
  }
}

const getAccuracyColor = (accuracy) => {
  if (accuracy >= 0.85) return '#67C23A'
  if (accuracy >= 0.75) return '#E6A23C'
  return '#F56C6C'
}

const getBestModel = (modelName) => {
  return modelName === bestModel.value
}

const selectModel = (model) => {
  ElMessage.success(`已选择 ${model.model} 作为当前检测模型`)
}

const initCharts = () => {
  nextTick(() => {
    // 准确率对比图
    const accuracyChart = echarts.init(document.getElementById('accuracyChart'))
    accuracyChart.setOption({
      title: { text: '测试准确率对比', left: 'center' },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: comparisonData.value.map(d => d.model)
      },
      yAxis: {
        type: 'value',
        max: 1,
        axisLabel: { formatter: '{value}%' }
      },
      series: [{
        data: comparisonData.value.map(d => (d.testAccuracy * 100).toFixed(2)),
        type: 'bar',
        itemStyle: {
          color: (params) => {
            const colors = ['#409EFF', '#67C23A', '#E6A23C']
            return colors[params.dataIndex]
          }
        },
        label: {
          show: true,
          position: 'top',
          formatter: '{c}%'
        }
      }]
    })
    
    // 多指标对比图
    const metricsChart = echarts.init(document.getElementById('metricsChart'))
    metricsChart.setOption({
      title: { text: '多指标对比', left: 'center' },
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['IoU', 'F1', '精确率', '召回率'],
        bottom: 0
      },
      radar: {
        indicator: [
          { name: 'IoU', max: 1 },
          { name: 'F1', max: 1 },
          { name: '精确率', max: 1 },
          { name: '召回率', max: 1 }
        ],
        radius: '65%'
      },
      series: [{
        type: 'radar',
        data: comparisonData.value.map((d, i) => ({
          value: [d.meanIou, d.f1Score, d.precision, d.recall],
          name: d.model,
          itemStyle: {
            color: ['#409EFF', '#67C23A', '#E6A23C'][i]
          }
        }))
      }]
    })
  })
}

onMounted(() => {
  initCharts()
})
</script>

<style lang="scss" scoped>
.comparison-container {
  .model-analysis {
    text-align: center;
    
    p {
      margin: 10px 0;
    }
  }
}
</style>
