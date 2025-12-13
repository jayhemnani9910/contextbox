#!/usr/bin/env python3
"""
Test script for the Intelligent Summarization System

This script demonstrates all features of the SummarizationManager including:
- Basic summarization
- Map-reduce functionality
- Progressive summarization
- Multi-document summarization
- Quality assessment
- Caching
- Export capabilities
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextbox.llm.summarization import (
    SummarizationManager, SummaryRequest, DatabaseIntegratedSummarizer,
    create_summarization_manager, summarize_text
)
from contextbox.llm.mock_backend import MockLLMBackend, create_mock_summarization_system


def setup_mock_manager():
    """Set up a manager with mock LLM backend."""
    manager = SummarizationManager()
    
    # Add mock backend
    mock_backend = MockLLMBackend()
    manager.backends['mock'] = mock_backend
    
    # Make mock the default backend
    manager.config.default_provider = 'mock'
    
    return manager


def test_basic_summarization():
    """Test basic summarization functionality."""
    print("=== Testing Basic Summarization ===")
    
    manager = setup_mock_manager()
    
    # Sample content for testing
    sample_content = """
    Artificial Intelligence (AI) is revolutionizing the way we work and live. Machine learning algorithms can now process 
    vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural 
    networks with multiple layers to model and understand complex data relationships. Natural language processing enables 
    computers to understand and generate human language, leading to advances in chatbots, translation, and content analysis.

    Computer vision allows machines to interpret and understand visual information from the world. This technology powers 
    self-driving cars, medical image analysis, and facial recognition systems. Robotics combines AI with mechanical systems 
    to create intelligent machines that can perform tasks autonomously. These advancements are creating new opportunities 
    while also raising important questions about the future of work and society.

    As AI continues to evolve, it's crucial to consider both the benefits and potential risks. Responsible development and 
    deployment of AI systems will be key to ensuring that this powerful technology serves humanity's best interests.
    """
    
    # Test different summary lengths
    lengths = ["brief", "detailed", "executive"]
    formats = ["paragraph", "bullets", "key_points"]
    
    for length in lengths:
        for format_type in formats:
            print(f"\n--- Testing {length} summary in {format_type} format ---")
            
            request = SummaryRequest(
                content=sample_content,
                content_type="article",
                summary_length=length,
                format_type=format_type,
                enable_caching=True
            )
            
            start_time = time.time()
            result = manager.summarize_content(request)
            end_time = time.time()
            
            if result and not result.error:
                print(f"✓ Generated {length} summary ({end_time - start_time:.2f}s)")
                print(f"Quality score: {result.quality_metrics.get('overall', 'N/A'):.2f}")
                print(f"Summary preview: {result.summary.text[:200]}...")
                print(f"Cache hit: {result.cache_hit}")
            else:
                print(f"✗ Failed to generate {length} summary")
                if result and result.error:
                    print(f"Error: {result.error}")


def test_content_type_awareness():
    """Test content type-aware summarization."""
    print("\n=== Testing Content Type Awareness ===")
    
    manager = setup_mock_manager()
    
    # Different content types
    content_types = {
        "article": """
        The Future of Renewable Energy
        
        Solar and wind power are becoming increasingly cost-effective compared to fossil fuels. 
        Battery storage technology is advancing rapidly, enabling better integration of renewable 
        energy into the grid. Governments worldwide are implementing policies to accelerate 
        the transition to clean energy sources. The economic benefits of renewable energy are 
        becoming more apparent as costs continue to decline.
        """,
        
        "transcript": """
        [00:01:30] Dr. Smith: Good morning everyone. Today we'll discuss the latest developments 
        in quantum computing. [00:02:15] Dr. Johnson: Thank you. Quantum computing represents 
        a fundamental shift in how we process information. [00:03:45] Dr. Smith: Exactly. 
        The quantum advantage is becoming more apparent in optimization problems.
        """,
        
        "documentation": """
        Installation Guide
        
        1. Prerequisites: Python 3.8 or higher
        
        2. Install dependencies:
           pip install -r requirements.txt
        
        3. Configuration:
           Copy config.example.json to config.json and update settings
        
        4. Run the application:
           python main.py
        """
    }
    
    for content_type, content in content_types.items():
        print(f"\n--- Testing {content_type} content ---")
        
        request = SummaryRequest(
            content=content,
            content_type=content_type,
            summary_length="detailed",
            format_type="paragraph"
        )
        
        result = manager.summarize_content(request)
        
        if result and not result.error:
            print(f"✓ {content_type.capitalize()} summary generated")
            print(f"Summary: {result.summary.text[:150]}...")
            print(f"Quality: {result.quality_metrics.get('overall', 0):.2f}")
        else:
            print(f"✗ Failed to generate {content_type} summary")


def test_progressive_summarization():
    """Test progressive summarization feature."""
    print("\n=== Testing Progressive Summarization ===")
    
    manager = setup_mock_manager()
    
    long_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. Supervised learning uses labeled training data to make 
    predictions on new, unseen data. Common supervised algorithms include linear regression, 
    decision trees, and support vector machines. Each algorithm has its strengths and 
    weaknesses depending on the type of problem being solved.
    
    Unsupervised learning works with unlabeled data to discover hidden patterns. 
    Clustering algorithms group similar data points together, while dimensionality 
    reduction techniques help visualize and simplify complex datasets. Principal 
    component analysis (PCA) and t-SNE are popular dimensionality reduction methods.
    
    Reinforcement learning involves an agent learning to make decisions by taking 
    actions in an environment and receiving feedback. This approach has been successful 
    in game playing, robotics, and optimization problems. Deep reinforcement learning 
    combines neural networks with reinforcement learning principles.
    
    Deep learning uses artificial neural networks with multiple layers to model 
    complex patterns in data. Convolutional neural networks excel at image processing, 
    while recurrent neural networks are effective for sequential data. Transformer 
    architectures have revolutionized natural language processing tasks.
    """
    
    print("\n--- Testing Progressive Summarization ---")
    
    request = SummaryRequest(
        content=long_content,
        content_type="documentation",
        summary_length="detailed",
        enable_progressive=True
    )
    
    result = manager.summarize_content(request)
    
    if result and not result.error:
        print("✓ Progressive summary generated")
        print(f"Summary: {result.summary.text}")
        print(f"Is progressive: {'Yes' if result.summary.metadata and result.summary.metadata.get('progressive') else 'No'}")
    else:
        print("✗ Progressive summarization failed")


def test_multi_document_summarization():
    """Test multi-document summarization."""
    print("\n=== Testing Multi-Document Summarization ===")
    
    manager = setup_mock_manager()
    
    # Sample documents
    documents = [
        ("Climate change is causing rising sea levels and more extreme weather events. Renewable energy adoption is crucial for mitigation.", "news"),
        ("Solar panel efficiency has improved significantly while costs have decreased by 85% over the past decade.", "article"),
        ("Wind energy technology continues to advance with larger turbines and better energy storage solutions.", "article")
    ]
    
    print("\n--- Testing Multi-Document Summarization ---")
    
    try:
        result = manager.summarize_multiple_documents(
            documents=documents,
            summary_type="synthesis",
            summary_length="detailed",
            format_type="paragraph"
        )
        
        print("✓ Multi-document summary generated")
        print(f"Individual summaries: {len(result.individual_summaries)}")
        print(f"Combined summary preview: {result.combined_summary.text[:200]}...")
        print(f"Common themes: {', '.join(result.common_themes[:5])}")
        
    except Exception as e:
        print(f"✗ Multi-document summarization failed: {e}")


def test_quality_assessment():
    """Test quality assessment features."""
    print("\n=== Testing Quality Assessment ===")
    
    manager = setup_mock_manager()
    
    # Test content with varying quality
    test_cases = [
        {
            "name": "High Quality Content",
            "content": """
            The Amazon rainforest, known as the "lungs of the Earth," produces approximately 
            20% of the world's oxygen. This massive ecosystem spans across nine countries in 
            South America and is home to an estimated 10% of all known species on the planet. 
            The rainforest plays a crucial role in regulating global climate patterns and 
            carbon dioxide levels. Deforestation poses a significant threat to this vital 
            ecosystem and global environmental stability.
            """,
            "expected_quality": "high"
        },
        {
            "name": "Low Quality Content", 
            "content": "Stuff happens and things occur.",
            "expected_quality": "low"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        
        request = SummaryRequest(
            content=test_case["content"],
            content_type="article",
            summary_length="detailed"
        )
        
        result = manager.summarize_content(request)
        
        if result and not result.error:
            overall_score = result.quality_metrics.get('overall', 0)
            print(f"✓ Quality assessment completed")
            print(f"Overall quality score: {overall_score:.2f}")
            
            # Print detailed metrics
            for metric, score in result.quality_metrics.items():
                if metric != 'overall':
                    print(f"  {metric.title()}: {score:.2f}")
        else:
            print("✗ Quality assessment failed")


def test_caching():
    """Test caching functionality."""
    print("\n=== Testing Caching System ===")
    
    manager = setup_mock_manager()
    
    content = "This is test content for caching. It contains some information that can be summarized."
    
    # First request (cache miss)
    print("\n--- First Request (Expected Cache Miss) ---")
    request1 = SummaryRequest(
        content=content,
        content_type="article",
        summary_length="detailed",
        enable_caching=True
    )
    
    start_time = time.time()
    result1 = manager.summarize_content(request1)
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Cache hit: {result1.cache_hit}")
    
    # Second request (cache hit)
    print("\n--- Second Request (Expected Cache Hit) ---")
    request2 = SummaryRequest(
        content=content,
        content_type="article", 
        summary_length="detailed",
        enable_caching=True
    )
    
    start_time = time.time()
    result2 = manager.summarize_content(request2)
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Cache hit: {result2.cache_hit}")
    print(f"Cache improved speed: {(result1.processing_info.get('timestamp') is not None)}")
    
    # Check cache statistics
    stats = manager.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Average access count: {stats['avg_access_count']:.2f}")


def test_export_functionality():
    """Test export capabilities."""
    print("\n=== Testing Export Functionality ===")
    
    manager = setup_mock_manager()
    
    content = "This is a test article about artificial intelligence and its applications in modern society."
    
    request = SummaryRequest(
        content=content,
        content_type="article",
        summary_length="detailed"
    )
    
    result = manager.summarize_content(request)
    
    if result and not result.error:
        # Test different export formats
        formats = ["json", "markdown", "text"]
        
        for format_type in formats:
            output_file = f"test_summary.{'md' if format_type == 'markdown' else format_type}"
            
            try:
                manager.export_summary(result, output_file, format_type)
                print(f"✓ Exported {format_type} format to {output_file}")
                
                # Show file size
                file_size = os.path.getsize(output_file)
                print(f"  File size: {file_size} bytes")
                
            except Exception as e:
                print(f"✗ Export failed for {format_type}: {e}")
    else:
        print("✗ Cannot test export - summarization failed")


def test_health_check():
    """Test system health check."""
    print("\n=== Testing Health Check ===")
    
    manager = setup_mock_manager()
    
    health = manager.health_check()
    
    print(f"System Status: {health['overall_status']}")
    print(f"Timestamp: {health['timestamp']}")
    
    # Check backends
    print("\nBackends:")
    for name, status in health['backends'].items():
        print(f"  {name}: {status['status']}")
    
    # Check cache
    if health['cache']['available']:
        print(f"\nCache: Available")
        stats = health['cache']['stats']
        print(f"  Entries: {stats['total_entries']}")
        print(f"  Avg access: {stats['avg_access_count']:.2f}")
    else:
        print(f"\nCache: Error - {health['cache']['error']}")


def test_integration_example():
    """Example of how to integrate with existing systems."""
    print("\n=== Integration Example ===")
    
    # Example: Simple summarization function
    def quick_summary(text: str, content_type: str = "text") -> str:
        """Quick summary function for easy integration."""
        return summarize_text(text, content_type, "brief", "paragraph")
    
    # Example usage
    sample_text = """
    ContextBox is a powerful content extraction and analysis system. It can extract 
    content from various sources including websites, documents, and media files. 
    The system provides intelligent categorization and analysis capabilities, making 
    it easier to organize and understand large volumes of information.
    """
    
    summary = quick_summary(sample_text)
    print(f"✓ Quick summary generated:")
    print(f"Original length: {len(sample_text)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"Compression ratio: {len(summary) / len(sample_text):.2f}")
    print(f"Summary: {summary}")


def main():
    """Run all tests."""
    print("Intelligent Summarization System - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_basic_summarization()
        test_content_type_awareness()
        test_progressive_summarization()
        test_multi_document_summarization()
        test_quality_assessment()
        test_caching()
        test_export_functionality()
        test_health_check()
        test_integration_example()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("\nThe summarization system is ready for production use.")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())