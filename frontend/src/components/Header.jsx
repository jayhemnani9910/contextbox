import React from 'react'
import { Link } from 'react-router-dom'
import { Github, Box } from 'lucide-react'

function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 bg-white border-b border-gray-200 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Box className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">ContextBox</span>
          </Link>

          <nav className="flex items-center space-x-6">
            <Link to="/installation" className="text-gray-600 hover:text-gray-900">
              Installation
            </Link>
            <Link to="/commands" className="text-gray-600 hover:text-gray-900">
              Commands
            </Link>
            <Link to="/api" className="text-gray-600 hover:text-gray-900">
              API
            </Link>
            <Link to="/demo" className="text-gray-600 hover:text-gray-900">
              Demo
            </Link>
            <a
              href="https://github.com/jayhemnani9910/contextbox"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900"
            >
              <Github className="h-5 w-5" />
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header
