import React from 'react'
import { Link } from 'react-router-dom'
import { Camera, Search, MessageSquare, FileText, Zap, Shield } from 'lucide-react'

const features = [
  {
    name: 'Screenshot Capture',
    description: 'Capture screenshots with automatic OCR text extraction.',
    icon: Camera,
  },
  {
    name: 'Semantic Search',
    description: 'Search across all your captured contexts intelligently.',
    icon: Search,
  },
  {
    name: 'AI-Powered Q&A',
    description: 'Ask questions about your captured context using GitHub Models.',
    icon: MessageSquare,
  },
  {
    name: 'Web Extraction',
    description: 'Extract content from web pages, Wikipedia, and YouTube.',
    icon: FileText,
  },
  {
    name: 'Fast & Efficient',
    description: 'Lightweight CLI with beautiful Rich terminal formatting.',
    icon: Zap,
  },
  {
    name: 'Privacy First',
    description: 'All data stored locally. Your context stays on your machine.',
    icon: Shield,
  },
]

function Home() {
  return (
    <div className="pt-8">
      {/* Hero */}
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          AI-Powered Context Capture
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
          Capture, organize, and query your digital context with powerful AI assistance.
          Free LLM access via GitHub Models.
        </p>
        <div className="flex justify-center space-x-4">
          <Link
            to="/installation"
            className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition"
          >
            Get Started
          </Link>
          <Link
            to="/demo"
            className="bg-gray-100 text-gray-900 px-6 py-3 rounded-lg font-medium hover:bg-gray-200 transition"
          >
            Live Demo
          </Link>
        </div>
      </div>

      {/* Quick Install */}
      <div className="bg-gray-900 text-white rounded-lg p-6 mb-16">
        <div className="text-center">
          <p className="text-gray-400 mb-2">Install with pip</p>
          <code className="text-lg font-mono">pip install contextbox</code>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {features.map((feature) => (
          <div key={feature.name} className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <feature.icon className="h-8 w-8 text-blue-600 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.name}</h3>
            <p className="text-gray-600">{feature.description}</p>
          </div>
        ))}
      </div>

      {/* Quick Example */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Example</h2>
        <div className="bg-gray-900 text-white rounded-lg p-6 font-mono text-sm overflow-x-auto">
          <pre>{`# Capture your screen
$ contextbox capture
✓ Screenshot captured
✓ Text extracted (1,234 chars)
✓ Context stored: ctx_abc123

# Ask questions about it
$ contextbox ask "What was I working on?"
Based on your captured context, you were working on...

# Search your contexts
$ contextbox search "API documentation"
Found 3 matching contexts...`}</pre>
        </div>
      </div>
    </div>
  )
}

export default Home
