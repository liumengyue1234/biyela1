<template>
  <div class="app-container">
    <el-container>
      <!-- 侧边栏 -->
      <el-aside width="200px">
        <div class="logo">
          <h2>松材线虫检测</h2>
        </div>
        <el-menu
          :default-active="$route.path"
          router
          background-color="#304156"
          text-color="#bfcbd9"
          active-text-color="#409EFF"
        >
          <el-menu-item index="/">
            <el-icon><HomeFilled /></el-icon>
            <span>首页</span>
          </el-menu-item>
          <el-menu-item index="/detect">
            <el-icon><UploadFilled /></el-icon>
            <span>图像检测</span>
          </el-menu-item>
          <el-menu-item index="/training">
            <el-icon><DataLine /></el-icon>
            <span>模型训练</span>
          </el-menu-item>
          <el-menu-item index="/history">
            <el-icon><Clock /></el-icon>
            <span>检测记录</span>
          </el-menu-item>
          <el-menu-item index="/comparison">
            <el-icon><TrendCharts /></el-icon>
            <span>模型对比</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-container>
        <!-- 顶部导航 -->
        <el-header>
          <div class="header-content">
            <h1>{{ pageTitle }}</h1>
            <div class="header-actions">
              <el-button type="primary" @click="checkSystemStatus">
                系统状态
              </el-button>
            </div>
          </div>
        </el-header>

        <!-- 主内容区 -->
        <el-main>
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { HomeFilled, UploadFilled, DataLine, Clock, TrendCharts } from '@element-plus/icons-vue'

const route = useRoute()

const pageTitle = computed(() => {
  const titles = {
    '/': '松材线虫病检测系统',
    '/detect': 'CT图像检测',
    '/training': '模型训练',
    '/history': '检测记录',
    '/comparison': '模型对比'
  }
  return titles[route.path] || '松材线虫病检测系统'
})

const checkSystemStatus = () => {
  ElMessage({
    message: '系统运行正常！',
    type: 'success'
  })
}
</script>

<style lang="scss" scoped>
.app-container {
  height: 100vh;
  
  .el-aside {
    background-color: #304156;
    
    .logo {
      height: 60px;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: #2b3a4a;
      
      h2 {
        color: #fff;
        font-size: 16px;
        margin: 0;
      }
    }
  }
  
  .el-header {
    background-color: #fff;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    display: flex;
    align-items: center;
    
    .header-content {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      
      h1 {
        font-size: 18px;
        color: #333;
        margin: 0;
      }
    }
  }
  
  .el-main {
    background-color: #f0f2f5;
    padding: 20px;
  }
}
</style>
