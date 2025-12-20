import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Home from './pages/Home'
import Installation from './pages/Installation'
import Commands from './pages/Commands'
import ApiReference from './pages/ApiReference'
import Demo from './pages/Demo'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-8 ml-64">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/installation" element={<Installation />} />
            <Route path="/commands" element={<Commands />} />
            <Route path="/api" element={<ApiReference />} />
            <Route path="/demo" element={<Demo />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
