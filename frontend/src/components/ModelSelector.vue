<template>
  <div class="model-selector">
    <el-card class="selector-card">
      <template #header>
        <div class="card-header">
          <span>选择检测模型</span>
        </div>
      </template>

      <el-radio-group v-model="selectedModel" @change="handleModelChange">
        <el-radio
          v-for="model in models"
          :key="model.value"
          :label="model.value"
          border
          class="model-radio"
        >
          <div class="model-content">
            <div class="model-name">
              <i :class="model.icon"></i>
              {{ model.label }}
            </div>
            <div class="model-desc">{{ model.description }}</div>
            <div class="model-meta">
              <span class="model-accuracy">准确率: {{ model.accuracy }}</span>
              <span class="model-speed">速度: {{ model.speed }}</span>
            </div>
          </div>
        </el-radio>
      </el-radio-group>

      <!-- 参数调整 -->
      <div class="params-adjust">
        <h4>参数设置</h4>
        <el-form label-width="100px" size="small">
          <el-form-item label="置信度阈值">
            <el-slider
              v-model="confidenceThreshold"
              :min="0.1"
              :max="0.95"
              :step="0.05"
              :format-tooltip="(val) => (val * 100).toFixed(0) + '%'"
              @change="handleParamChange"
            />
            <span class="threshold-value">{{ (confidenceThreshold * 100).toFixed(0) }}%</span>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
  </div>
</template>

<script>
import { defineComponent, ref, watch } from 'vue'

export default defineComponent({
  name: 'ModelSelector',
  props: {
    model: {
      type: String,
      default: 'UNET'
    },
    threshold: {
      type: Number,
      default: 0.5
    }
  },
  emits: ['update:model', 'update:threshold', 'change'],
  setup(props, { emit }) {
    const models = [
      {
        value: 'UNET',
        label: '改进 U-Net',
        icon: 'el-icon-picture-outline',
        description: '语义分割模型，适合精细边界检测',
        accuracy: '89.5%',
        speed: '中等'
      },
      {
        value: 'MASK_RCNN',
        label: 'Mask R-CNN',
        icon: 'el-icon-menu',
        description: '实例分割模型，可区分不同目标',
        accuracy: '87.3%',
        speed: '较慢'
      },
      {
        value: 'YOLOv8',
        label: 'YOLOv8',
        icon: 'el-icon-s-grid',
        description: 'Ultralytics经典版本，精度与速度平衡',
        accuracy: '85.8%',
        speed: '快速'
      },
      {
        value: 'YOLO11',
        label: 'YOLO11',
        icon: 'el-icon-s-grid',
        description: '2024年新版，增强特征提取能力',
        accuracy: '86.5%',
        speed: '快速'
      },
      {
        value: 'YOLOv10',
        label: 'YOLOv10',
        icon: 'el-icon-s-grid',
        description: '清华大学端到端版本，无需NMS后处理',
        accuracy: '84.2%',
        speed: '最快'
      }
    ]

    const selectedModel = ref(props.model)
    const confidenceThreshold = ref(props.threshold)

    watch(() => props.model, (val) => {
      selectedModel.value = val
    })

    watch(() => props.threshold, (val) => {
      confidenceThreshold.value = val
    })

    const handleModelChange = (val) => {
      emit('update:model', val)
      emit('change', { model: val, threshold: confidenceThreshold.value })
    }

    const handleParamChange = (val) => {
      emit('update:threshold', val)
      emit('change', { model: selectedModel.value, threshold: val })
    }

    return {
      models,
      selectedModel,
      confidenceThreshold,
      handleModelChange,
      handleParamChange
    }
  }
})
</script>

<style scoped>
.model-selector {
  width: 100%;
}

.selector-card {
  border-radius: 8px;
}

.card-header {
  font-weight: bold;
}

.model-radio {
  width: 100%;
  margin-bottom: 12px;
  margin-left: 0 !important;
}

.model-content {
  padding: 10px 0;
  text-align: left;
}

.model-name {
  font-size: 16px;
  font-weight: bold;
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-name i {
  font-size: 20px;
  color: #409eff;
}

.model-desc {
  font-size: 13px;
  color: #606266;
  margin-top: 5px;
}

.model-meta {
  display: flex;
  gap: 15px;
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}

.model-accuracy {
  color: #67c23a;
}

.model-speed {
  color: #e6a23c;
}

.params-adjust {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}

.params-adjust h4 {
  margin-bottom: 15px;
  color: #303133;
}

.threshold-value {
  margin-left: 15px;
  font-weight: bold;
  color: #409eff;
}
</style>
