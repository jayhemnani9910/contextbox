import React from 'react'
import CodeBlock from '../components/CodeBlock'

function Installation() {
  return (
    <div className="pt-8 max-w-3xl">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Installation</h1>

      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Requirements</h2>
        <ul className="list-disc list-inside text-gray-600 space-y-2">
          <li>Python 3.9 or higher</li>
          <li>pip package manager</li>
          <li>For OCR: Tesseract OCR engine</li>
        </ul>
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Using pip</h2>
        <CodeBlock code="pip install contextbox" />
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">From Source</h2>
        <CodeBlock
          code={`git clone https://github.com/jayhemnani9910/contextbox.git
cd contextbox
pip install -e ".[all]"`}
        />
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Optional Dependencies</h2>
        <div className="space-y-4">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">LLM Features (GitHub Models)</h3>
            <CodeBlock code='pip install contextbox[llm]' />
          </div>
          <div>
            <h3 className="font-medium text-gray-900 mb-2">OCR Support</h3>
            <CodeBlock code='pip install contextbox[ocr]' />
          </div>
          <div>
            <h3 className="font-medium text-gray-900 mb-2">YouTube Extraction</h3>
            <CodeBlock code='pip install contextbox[youtube]' />
          </div>
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Everything</h3>
            <CodeBlock code='pip install contextbox[all]' />
          </div>
        </div>
      </section>

      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Setup GitHub Token</h2>
        <p className="text-gray-600 mb-4">
          To use AI features (Q&A, summarization), set your GitHub token:
        </p>
        <CodeBlock code='export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"' />
        <p className="text-gray-500 text-sm mt-2">
          Get a token from GitHub Settings → Developer settings → Personal access tokens
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Verify Installation</h2>
        <CodeBlock
          code={`contextbox --version
# ContextBox CLI v1.0.0`}
        />
      </section>
    </div>
  )
}

export default Installation
