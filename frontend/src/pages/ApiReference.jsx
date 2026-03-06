import React from 'react'
import CodeBlock from '../components/CodeBlock'

function ApiReference() {
  return (
    <div className="pt-8 max-w-4xl">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">API Reference</h1>
      <p className="text-gray-600 mb-8">
        Use ContextBox programmatically in your Python applications.
      </p>

      {/* ContextBox Class */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">ContextBox Class</h2>
        <p className="text-gray-600 mb-4">Main class for context capture and management.</p>

        <CodeBlock
          code={`from contextbox import ContextBox

# Initialize
app = ContextBox(config={})

# Capture current screen
context = app.capture()

# Store context
context_id = app.store_context(context)

# Retrieve context
retrieved = app.get_context(context_id)

# Search contexts
results = app.search("keyword")`}
        />
      </section>

      {/* LLM Backend */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">LLM Backend</h2>
        <p className="text-gray-600 mb-4">
          Use GitHub Models for AI-powered features.
        </p>

        <CodeBlock
          code={`from contextbox.llm import (
    create_github_models_backend,
    ChatRequest,
    quick_chat
)

# Quick one-liner
response = await quick_chat("Summarize this text")

# Full control
backend = create_github_models_backend(
    default_model="gpt-4o-mini"
)

async with backend:
    request = ChatRequest.from_text(
        text="Your prompt here",
        model="gpt-4o-mini",
        provider="github_models",
        system_prompt="You are a helpful assistant."
    )
    response = await backend.chat_completion(request)
    print(response.content)`}
        />
      </section>

      {/* Available Models */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Available Models</h2>
        <div className="bg-gray-50 rounded-lg overflow-hidden">
          <table className="min-w-full">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Model</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Context Length</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              <tr>
                <td className="px-4 py-3 font-mono text-sm">gpt-4o</td>
                <td className="px-4 py-3 text-sm">128K</td>
                <td className="px-4 py-3 text-sm text-gray-600">Most capable GPT-4 model</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-mono text-sm">gpt-4o-mini</td>
                <td className="px-4 py-3 text-sm">128K</td>
                <td className="px-4 py-3 text-sm text-gray-600">Fast and efficient (default)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-mono text-sm">meta-llama-3.1-405b-instruct</td>
                <td className="px-4 py-3 text-sm">128K</td>
                <td className="px-4 py-3 text-sm text-gray-600">Meta's largest Llama</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-mono text-sm">meta-llama-3.1-70b-instruct</td>
                <td className="px-4 py-3 text-sm">128K</td>
                <td className="px-4 py-3 text-sm text-gray-600">Meta's Llama 70B</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-mono text-sm">mistral-large</td>
                <td className="px-4 py-3 text-sm">32K</td>
                <td className="px-4 py-3 text-sm text-gray-600">Mistral's flagship model</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Extractors */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Content Extractors</h2>
        <p className="text-gray-600 mb-4">
          Extract content from various sources.
        </p>

        <CodeBlock
          code={`from contextbox.extractors import (
    WebPageExtractor,
    WikipediaExtractor,
    YouTubeExtractor
)

# Web page extraction
extractor = WebPageExtractor()
content = extractor.extract("https://example.com")

# Wikipedia
wiki = WikipediaExtractor()
article = wiki.extract("Python (programming language)")

# YouTube
yt = YouTubeExtractor()
transcript = yt.extract("https://youtube.com/watch?v=...")`}
        />
      </section>

      {/* Response Types */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">Response Types</h2>

        <h3 className="text-lg font-medium text-gray-900 mb-2">LLMResponse</h3>
        <CodeBlock
          code={`@dataclass
class LLMResponse:
    content: str          # Generated text
    model: str            # Model used
    provider: str         # Provider name
    usage: TokenUsage     # Token counts
    cost: CostInfo        # Cost information (free for GitHub Models)
    finish_reason: str    # Why generation stopped
    metadata: dict        # Additional info`}
        />

        <h3 className="text-lg font-medium text-gray-900 mt-6 mb-2">TokenUsage</h3>
        <CodeBlock
          code={`@dataclass
class TokenUsage:
    prompt_tokens: int       # Input tokens
    completion_tokens: int   # Output tokens
    total_tokens: int        # Total tokens`}
        />
      </section>
    </div>
  )
}

export default ApiReference
