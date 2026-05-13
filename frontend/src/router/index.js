import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue')
  },
  {
    path: '/detect',
    name: 'Detect',
    component: () => import('../views/Detect.vue')
  },
  {
    path: '/training',
    name: 'Training',
    component: () => import('../views/Training.vue')
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('../views/History.vue')
  },
  {
    path: '/comparison',
    name: 'Comparison',
    component: () => import('../views/Comparison.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
