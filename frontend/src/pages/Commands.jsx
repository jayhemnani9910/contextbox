import React from 'react'
import CodeBlock from '../components/CodeBlock'

const commands = [
  {
    name: 'capture',
    description: 'Capture screenshot and extract context from current screen',
    usage: 'contextbox capture [OPTIONS]',
    options: [
      { flag: '-o, --output PATH', desc: 'Output file for capture results' },
      { flag: '-a, --artifact-dir PATH', desc: 'Directory for artifacts (default: artifacts)' },
      { flag: '--no-screenshot', desc: 'Skip taking screenshot' },
      { flag: '--extract-text', desc: 'Extract text content' },
      { flag: '--extract-urls', desc: 'Extract URLs from content' },
    ],
    example: 'contextbox capture -o my_capture.json',
  },
  {
    name: 'list',
    description: 'List all stored contexts with display options',
    usage: 'contextbox list [OPTIONS]',
    options: [
      { flag: '--limit N', desc: 'Maximum contexts to show (default: 20)' },
      { flag: '--format TYPE', desc: 'Output format: table, json, brief, tree' },
      { flag: '--sort BY', desc: 'Sort by: timestamp, platform, status' },
    ],
    example: 'contextbox list --format table --limit 10',
  },
  {
    name: 'search',
    description: 'Search through stored contexts',
    usage: 'contextbox search QUERY [OPTIONS]',
    options: [
      { flag: '--context-type TYPE', desc: 'Filter by: all, text, urls, screenshots' },
      { flag: '--limit N', desc: 'Maximum results (default: 10)' },
      { flag: '--fuzzy', desc: 'Use fuzzy matching' },
    ],
    example: 'contextbox search "API documentation" --fuzzy',
  },
  {
    name: 'ask',
    description: 'Ask questions about captured context using AI',
    usage: 'contextbox ask QUESTION [OPTIONS]',
    options: [
      { flag: '--context-id ID', desc: 'Specific context to ask about' },
      { flag: '--all-contexts', desc: 'Search across all contexts' },
      { flag: '--model NAME', desc: 'LLM model to use' },
    ],
    example: 'contextbox ask "What was I working on yesterday?"',
  },
  {
    name: 'summarize',
    description: 'Generate intelligent summaries of captured contexts',
    usage: 'contextbox summarize [OPTIONS]',
    options: [
      { flag: '--context-id ID', desc: 'Specific context to summarize' },
      { flag: '--all-contexts', desc: 'Summarize all contexts' },
      { flag: '--format TYPE', desc: 'Format: brief, detailed, bullets, executive' },
    ],
    example: 'contextbox summarize --all-contexts --format bullets',
  },
  {
    name: 'stats',
    description: 'Display database and application statistics',
    usage: 'contextbox stats [OPTIONS]',
    options: [
      { flag: '--detailed', desc: 'Show detailed statistics' },
      { flag: '--format TYPE', desc: 'Output format: table, json, markdown' },
    ],
    example: 'contextbox stats --detailed',
  },
  {
    name: 'export',
    description: 'Export contexts to various file formats',
    usage: 'contextbox export [OPTIONS]',
    options: [
      { flag: '--format TYPE', desc: 'Format: json, csv, txt, markdown' },
      { flag: '-o, --output PATH', desc: 'Output file path' },
      { flag: '--include-artifacts', desc: 'Include file artifacts' },
    ],
    example: 'contextbox export --format json -o backup.json',
  },
  {
    name: 'config',
    description: 'Configure API keys and application settings',
    usage: 'contextbox config [OPTIONS]',
    options: [
      { flag: '--api-key', desc: 'Configure API key for AI features' },
      { flag: '--view', desc: 'View current configuration' },
      { flag: '--reset', desc: 'Reset to defaults' },
    ],
    example: 'contextbox config --api-key',
  },
]

function Commands() {
  return (
    <div className="pt-8 max-w-4xl">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">CLI Commands</h1>
      <p className="text-gray-600 mb-8">
        ContextBox provides a powerful CLI for capturing and managing your digital context.
      </p>

      <div className="space-y-12">
        {commands.map((cmd) => (
          <section key={cmd.name} id={cmd.name} className="scroll-mt-20">
            <h2 className="text-2xl font-semibold text-gray-900 mb-2">
              <code className="bg-blue-50 text-blue-700 px-2 py-1 rounded">{cmd.name}</code>
            </h2>
            <p className="text-gray-600 mb-4">{cmd.description}</p>

            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-500 uppercase mb-2">Usage</h3>
              <CodeBlock code={cmd.usage} />
            </div>

            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-500 uppercase mb-2">Options</h3>
              <div className="bg-gray-50 rounded-lg overflow-hidden">
                <table className="min-w-full">
                  <tbody className="divide-y divide-gray-200">
                    {cmd.options.map((opt, i) => (
                      <tr key={i}>
                        <td className="px-4 py-2 font-mono text-sm text-gray-900">{opt.flag}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{opt.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-500 uppercase mb-2">Example</h3>
              <CodeBlock code={cmd.example} />
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}

export default Commands
