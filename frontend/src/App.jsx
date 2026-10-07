import React, { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Home from './pages/Home'
import Installation from './pages/Installation'
import Commands from './pages/Commands'
import ApiReference from './pages/ApiReference'
import Demo from './pages/Demo'

function App() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50">
      <Header menuOpen={menuOpen} onMenuToggle={() => setMenuOpen(!menuOpen)} />
      <div className="flex">
        <Sidebar open={menuOpen} onClose={() => setMenuOpen(false)} />
        <main className="flex-1 min-w-0 ml-0 lg:ml-64 p-4 sm:p-8 pt-20">
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
