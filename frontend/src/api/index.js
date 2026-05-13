import axios from 'axios'

const API_BASE_URL = 'http://localhost:8080/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  config => {
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// 检测相关API
export const detectionAPI = {
  // 上传图像并检测
  uploadAndDetect: (file, modelType = 'UNET', threshold = 0.5) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('modelType', modelType)
    formData.append('threshold', threshold)
    
    return axios.post(`${API_BASE_URL}/detection/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 120000
    })
  },
  
  // 使用指定路径检测
  detect: (imagePath, modelType = 'UNET', threshold = 0.5) => {
    return apiClient.post('/detection/detect', {
      imagePath,
      modelType,
      threshold
    })
  },
  
  // 获取检测记录列表
  getRecords: (page = 0, size = 10) => {
    return apiClient.get('/detection/records', { params: { page, size } })
  },
  
  // 获取检测记录详情
  getRecord: (id) => {
    return apiClient.get(`/detection/records/${id}`)
  },
  
  // 获取可用模型列表
  getModels: () => {
    return apiClient.get('/detection/models')
  }
}

// 训练相关API
export const trainingAPI = {
  // 开始训练
  startTraining: (params, configFile = null) => {
    const formData = new FormData()
    Object.keys(params).forEach(key => {
      formData.append(key, params[key])
    })
    if (configFile) {
      formData.append('configFile', configFile)
    }
    
    return axios.post(`${API_BASE_URL}/training/start`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 600000 // 10分钟超时
    })
  },
  
  // 获取训练记录列表
  getRecords: (page = 0, size = 10) => {
    return apiClient.get('/training/records', { params: { page, size } })
  },
  
  // 获取指定模型的训练历史
  getHistory: (modelType) => {
    return apiClient.get(`/training/history/${modelType}`)
  },
  
  // 获取训练记录详情
  getRecord: (id) => {
    return apiClient.get(`/training/records/${id}`)
  },
  
  // 获取模型对比数据
  getComparison: () => {
    return apiClient.get('/training/comparison')
  }
}

export default {
  detectionAPI,
  trainingAPI
}
