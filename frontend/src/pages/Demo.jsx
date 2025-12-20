import React, { useState } from 'react'
import { Play, Copy, Check } from 'lucide-react'

const sampleContexts = [
  {
    id: 'ctx_001',
    timestamp: '2024-01-15T10:30:00',
    platform: 'Linux',
    text: 'Working on the API documentation for the new authentication module. Need to add OAuth2 flow diagrams.',
    urls: ['https://docs.example.com/auth', 'https://oauth.net/2/'],
  },
  {
    id: 'ctx_002',
    timestamp: '2024-01-15T14:22:00',
    platform: 'Linux',
    text: 'Debugging the database connection timeout issue. Found that the pool size was too small for concurrent requests.',
    urls: ['https://stackoverflow.com/questions/12345'],
  },
  {
    id: 'ctx_003',
    timestamp: '2024-01-15T16:45:00',
    platform: 'Linux',
    text: 'Reviewed PR #42 for the new caching layer. Suggested using Redis instead of in-memory cache for production.',
    urls: ['https://github.com/example/repo/pull/42'],
  },
]

const demoResponses = {
  capture: {
    output: `✓ Screenshot captured
✓ Text extracted (847 chars)
✓ URLs found: 2
✓ Context stored: ctx_004

Context ID: ctx_004
Timestamp: 2024-01-15T18:30:00
Platform: Linux`,
    delay: 1500,
  },
  list: {
    output: `╭──────────────────────────────────────────────────────────────╮
│                      Stored Contexts                         │
├──────────┬─────────────────────┬──────────┬─────────┬────────┤
│ ID       │ Timestamp           │ Platform │ Text    │ URLs   │
├──────────┼─────────────────────┼──────────┼─────────┼────────┤
│ ctx_001  │ 2024-01-15 10:30:00 │ Linux    │ 156     │ 2      │
│ ctx_002  │ 2024-01-15 14:22:00 │ Linux    │ 132     │ 1      │
│ ctx_003  │ 2024-01-15 16:45:00 │ Linux    │ 118     │ 1      │
╰──────────┴─────────────────────┴──────────┴─────────┴────────╯

Total: 3 contexts`,
    delay: 800,
  },
  ask: {
    output: `🤔 Processing your question...

Based on your captured contexts, here's what I found:

Today you worked on several tasks:
1. **API Documentation** - Writing docs for the authentication module,
   including OAuth2 flow diagrams
2. **Database Debugging** - Fixed connection timeout issues by increasing
   the connection pool size
3. **Code Review** - Reviewed PR #42 and recommended Redis for caching

Your main focus areas were authentication, database optimization, and
code review activities.`,
    delay: 2500,
  },
  search: {
    output: `🔍 Searching for "database"...

╭──────────────────────────────────────────────────────────────╮
│                     Search Results                           │
├──────────┬───────┬───────────────────────────────────────────┤
│ ID       │ Score │ Preview                                   │
├──────────┼───────┼───────────────────────────────────────────┤
│ ctx_002  │ 0.95  │ Debugging the database connection time... │
╰──────────┴───────┴───────────────────────────────────────────╯

Found 1 matching context`,
    delay: 1200,
  },
}

function Demo() {
  const [command, setCommand] = useState('')
  const [output, setOutput] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [copied, setCopied] = useState(false)

  const runCommand = (cmd) => {
    setIsRunning(true)
    setCommand(cmd)
    setOutput('')

    const key = cmd.split(' ')[1] // Get command name
    const response = demoResponses[key]

    if (response) {
      setTimeout(() => {
        setOutput(response.output)
        setIsRunning(false)
      }, response.delay)
    } else {
      setTimeout(() => {
        setOutput(`Command not found: ${cmd}\nTry: capture, list, ask, search`)
        setIsRunning(false)
      }, 500)
    }
  }

  const copyCommand = (cmd) => {
    navigator.clipboard.writeText(cmd)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="pt-8 max-w-4xl">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Live Demo</h1>
      <p className="text-gray-600 mb-8">
        Try ContextBox commands interactively. This demo uses simulated data to show
        how the CLI works.
      </p>

      {/* Command Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <button
          onClick={() => runCommand('contextbox capture')}
          disabled={isRunning}
          className="flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition"
        >
          <Play className="w-4 h-4 mr-2" />
          capture
        </button>
        <button
          onClick={() => runCommand('contextbox list')}
          disabled={isRunning}
          className="flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
        >
          <Play className="w-4 h-4 mr-2" />
          list
        </button>
        <button
          onClick={() => runCommand('contextbox ask "What did I work on today?"')}
          disabled={isRunning}
          className="flex items-center justify-center px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition"
        >
          <Play className="w-4 h-4 mr-2" />
          ask
        </button>
        <button
          onClick={() => runCommand('contextbox search "database"')}
          disabled={isRunning}
          className="flex items-center justify-center px-4 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 transition"
        >
          <Play className="w-4 h-4 mr-2" />
          search
        </button>
      </div>

      {/* Terminal Output */}
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>
          <span className="text-gray-400 text-sm">Terminal</span>
          <button
            onClick={() => copyCommand(command)}
            className="text-gray-400 hover:text-white"
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
        <div className="p-4 font-mono text-sm min-h-[300px]">
          {command && (
            <div className="text-green-400 mb-4">
              $ {command}
              {isRunning && <span className="animate-pulse ml-1">▊</span>}
            </div>
          )}
          {output && (
            <pre className="text-gray-300 whitespace-pre-wrap">{output}</pre>
          )}
          {!command && (
            <div className="text-gray-500">
              Click a command button above to see it in action...
            </div>
          )}
        </div>
      </div>

      {/* Sample Data */}
      <div className="mt-12">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Sample Context Data</h2>
        <p className="text-gray-600 mb-4">
          This demo uses the following sample contexts:
        </p>
        <div className="space-y-4">
          {sampleContexts.map((ctx) => (
            <div key={ctx.id} className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <code className="text-sm bg-blue-50 text-blue-700 px-2 py-1 rounded">
                  {ctx.id}
                </code>
                <span className="text-sm text-gray-500">{ctx.timestamp}</span>
              </div>
              <p className="text-gray-700 mb-2">{ctx.text}</p>
              <div className="flex flex-wrap gap-2">
                {ctx.urls.map((url, i) => (
                  <span
                    key={i}
                    className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded"
                  >
                    {url}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Demo
