import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/contextbox/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
