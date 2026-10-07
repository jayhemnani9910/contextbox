import React from 'react'
import { NavLink } from 'react-router-dom'
import { Home, Download, Terminal, Code, Play } from 'lucide-react'

const navigation = [
  { name: 'Overview', href: '/', icon: Home },
  { name: 'Installation', href: '/installation', icon: Download },
  { name: 'CLI Commands', href: '/commands', icon: Terminal },
  { name: 'API Reference', href: '/api', icon: Code },
  { name: 'Live Demo', href: '/demo', icon: Play },
]

function Sidebar({ open, onClose }) {
  return (
    <>
      {open && (
        <div className="fixed inset-0 top-16 z-30 bg-black/30 lg:hidden" onClick={onClose} />
      )}
      <aside
        className={`${open ? 'block' : 'hidden'} lg:block fixed left-0 top-16 bottom-0 z-40 w-64 bg-white border-r border-gray-200 overflow-y-auto`}
      >
        <nav className="p-4 space-y-1">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center px-4 py-2 text-sm font-medium rounded-md ${
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <item.icon className="mr-3 h-5 w-5" />
              {item.name}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200 mt-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-900 mb-2">Quick Install</h4>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded block break-all">
              pip install git+https://github.com/jayhemnani9910/contextbox.git
            </code>
            <p className="text-xs text-gray-500 mt-2">Not on PyPI yet.</p>
          </div>
        </div>
      </aside>
    </>
  )
}

export default Sidebar
