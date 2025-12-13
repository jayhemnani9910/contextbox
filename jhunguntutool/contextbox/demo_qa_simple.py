#!/usr/bin/env python3
"""
ContextBox QA System Simple Demo

This script demonstrates the key features of the ContextBox Question-Answering System
without requiring user interaction.
"""

import sys
import os
import tempfile

# Add the contextbox module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from contextbox.database import ContextDatabase
from contextbox.llm import QASystem


def setup_demo_content(database):
    """Set up sample content for demonstration."""
    print("Setting up demo content...")
    
    # Create a capture
    capture_id = database.create_capture(
        source_window="Demo Content",
        notes="Sample content for QA system demonstration"
    )
    
    # Add various types of content
    content_items = [
        {
            'kind': 'wikipedia_content',
            'title': 'Python Programming Language',
            'text': """Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a scripting 
            or glue language to connect existing components together.""",
            'metadata': {'confidence': 0.9, 'source_type': 'wikipedia'}
        },
        {
            'kind': 'tutorial_content',
            'title': 'Installing Python on Ubuntu',
            'text': """To install Python 3 on Ubuntu:
            1. Open the terminal (Ctrl+Alt+T)
            2. Update package index: sudo apt update
            3. Install Python 3: sudo apt install python3
            4. Install pip: sudo apt install python3-pip
            5. Verify installation: python3 --version""",
            'metadata': {'confidence': 0.85, 'source_type': 'tutorial'}
        },
        {
            'kind': 'article_content',
            'title': 'Artificial Intelligence Overview',
            'text': """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to 
            human intelligence. AI is used in various applications including natural language processing, 
            computer vision, and expert systems. Machine learning is a subset of AI that enables computers 
            to learn without being explicitly programmed.""",
            'metadata': {'confidence': 0.8, 'source_type': 'article'}
        }
    ]
    
    # Add each content item to the database
    for item in content_items:
        database.create_artifact(
            capture_id=capture_id,
            kind=item['kind'],
            title=item['title'],
            text=item['text'],
            metadata=item['metadata']
        )
    
    print(f"✓ Added {len(content_items)} content items to database")
    return capture_id


def demo_qa_system():
    """Run the QA system demonstration."""
    print("=" * 80)
    print("CONTEXTBOX QUESTION-ANSWERING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'demo_contextbox.db')
    database = ContextDatabase({'db_path': db_path})
    
    try:
        # Setup demo content
        setup_demo_content(database)
        
        # Initialize QA system
        qa_system = QASystem(database)
        print("\n✓ QA System initialized")
        
        # Demo questions
        demo_questions = [
            "What is Python programming language?",
            "How do I install Python on Ubuntu?", 
            "What are the main differences between AI and machine learning?",
            "Compare artificial intelligence and programming languages",
            "List the steps to install Python"
        ]
        
        print(f"\nTesting {len(demo_questions)} different types of questions...")
        
        # Test different types of questions
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{'='*15} Question {i} {'='*15}")
            print(f"Q: {question}")
            print("-" * 50)
            
            # Get answer
            answer = qa_system.ask_question(question)
            
            # Display answer
            print(f"A: {answer.answer_text}")
            print(f"\n📊 Answer Details:")
            print(f"   • Type: {answer.question_type.value}")
            print(f"   • Confidence: {answer.confidence_level.value} ({answer.confidence_score:.2f})")
            print(f"   • Sources: {len(answer.sources)}")
            print(f"   • Processing time: {answer.processing_time:.3f}s")
            
            # Show sources
            if answer.sources:
                print(f"\n📚 Sources:")
                for j, source in enumerate(answer.sources[:2], 1):
                    title = source.title or f"{source.kind} #{source.artifact_id}"
                    print(f"   {j}. {title}")
            
            # Show follow-up suggestions
            if answer.follow_up_suggestions:
                print(f"\n💡 Related Questions:")
                for suggestion in answer.follow_up_suggestions[:2]:
                    print(f"   • {suggestion}")
        
        # Show system statistics
        print(f"\n{'='*20} System Statistics {'='*20}")
        system_stats = qa_system.get_qa_statistics()
        
        print(f"📊 Overall Performance:")
        perf_metrics = system_stats['performance_metrics']
        print(f"   • Total questions processed: {system_stats['system_stats']['total_questions']}")
        print(f"   • Success rate: {perf_metrics['success_rate']:.1%}")
        print(f"   • Average processing time: {perf_metrics['average_processing_time']:.3f}s")
        print(f"   • Database captures: {system_stats['database_stats']['total_captures']}")
        print(f"   • Database artifacts: {system_stats['database_stats']['total_artifacts']}")
        
        print(f"\n{'='*80}")
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        database.cleanup()
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up demo resources")


def main():
    """Main function."""
    demo_qa_system()


if __name__ == "__main__":
    main()