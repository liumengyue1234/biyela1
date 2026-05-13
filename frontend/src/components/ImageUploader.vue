<template>
  <div class="image-uploader">
    <el-card class="upload-card">
      <div
        class="upload-area"
        :class="{ 'is-dragover': isDragover }"
        @drop.prevent="handleDrop"
        @dragover.prevent="isDragover = true"
        @dragleave.prevent="isDragover = false"
        @click="triggerUpload"
      >
        <i class="el-icon-upload"></i>
        <div class="upload-text">
          <span>点击或拖拽图片到此处上传</span>
          <span class="upload-tip">支持 JPG、PNG 格式，单文件最大 10MB</span>
        </div>
        <input
          ref="fileInput"
          type="file"
          accept="image/jpeg,image/png"
          style="display: none"
          @change="handleFileChange"
        />
      </div>

      <!-- 图片预览 -->
      <div v-if="previewUrl" class="preview-container">
        <el-image
          :src="previewUrl"
          :preview-src-list="[previewUrl]"
          fit="contain"
          class="preview-image"
        />
        <el-button
          type="danger"
          size="small"
          icon="el-icon-delete"
          class="remove-btn"
          @click="removeImage"
        >
          移除
        </el-button>
      </div>

      <!-- 上传进度 -->
      <el-progress
        v-if="uploading"
        :percentage="uploadProgress"
        :status="uploadStatus"
        class="upload-progress"
      />
    </el-card>
  </div>
</template>

<script>
import { defineComponent, ref } from 'vue'
import { ElMessage } from 'element-plus'

export default defineComponent({
  name: 'ImageUploader',
  emits: ['upload-success', 'upload-error', 'file-change'],
  setup(props, { emit }) {
    const fileInput = ref(null)
    const previewUrl = ref('')
    const isDragover = ref(false)
    const uploading = ref(false)
    const uploadProgress = ref(0)
    const uploadStatus = ref('')
    const currentFile = ref(null)

    const triggerUpload = () => {
      fileInput.value?.click()
    }

    const validateFile = (file) => {
      const isImage = ['image/jpeg', 'image/png'].includes(file.type)
      const isLt10M = file.size / 1024 / 1024 < 10

      if (!isImage) {
        ElMessage.error('只能上传 JPG 或 PNG 格式的图片!')
        emit('upload-error', '文件格式错误')
        return false
      }
      if (!isLt10M) {
        ElMessage.error('图片大小不能超过 10MB!')
        emit('upload-error', '文件太大')
        return false
      }
      return true
    }

    const handleFileChange = (event) => {
      const file = event.target.files[0]
      if (file && validateFile(file)) {
        processFile(file)
      }
    }

    const handleDrop = (event) => {
      isDragover.value = false
      const file = event.dataTransfer.files[0]
      if (file && validateFile(file)) {
        processFile(file)
      }
    }

    const processFile = (file) => {
      currentFile.value = file
      previewUrl.value = URL.createObjectURL(file)
      emit('file-change', file)
    }

    const removeImage = () => {
      previewUrl.value = ''
      currentFile.value = null
      if (fileInput.value) {
        fileInput.value.value = ''
      }
      emit('file-change', null)
    }

    const getFile = () => currentFile.value

    return {
      fileInput,
      previewUrl,
      isDragover,
      uploading,
      uploadProgress,
      uploadStatus,
      triggerUpload,
      handleFileChange,
      handleDrop,
      removeImage,
      getFile
    }
  }
})
</script>

<style scoped>
.image-uploader {
  width: 100%;
}

.upload-card {
  border-radius: 8px;
}

.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  border: 2px dashed #dcdfe6;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s;
  background: #fafafa;
}

.upload-area:hover,
.upload-area.is-dragover {
  border-color: #409eff;
  background: #ecf5ff;
}

.upload-area i {
  font-size: 67px;
  color: #c0c4cc;
  margin-bottom: 16px;
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #606266;
}

.upload-tip {
  font-size: 12px;
  color: #909399;
}

.preview-container {
  margin-top: 20px;
  position: relative;
}

.preview-image {
  width: 100%;
  max-height: 400px;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.remove-btn {
  position: absolute;
  top: 10px;
  right: 10px;
}

.upload-progress {
  margin-top: 20px;
}
</style>
