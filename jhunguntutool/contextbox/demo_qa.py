#!/usr/bin/env python3
"""
ContextBox QA System Demo

This script demonstrates the key features of the ContextBox Question-Answering System.
Run this script to see the system in action with sample content and questions.
"""

import sys
import os
import tempfile

# Add the contextbox module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'contextbox'))

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
            or glue language to connect existing components together. Python's simple, easy to learn 
            syntax emphasizes readability and therefore reduces the cost of program maintenance.""",
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
            5. Verify installation: python3 --version
            6. Install virtual environment: sudo apt install python3-venv
            
            Common Python packages can be installed using pip:
            pip3 install requests numpy pandas matplotlib""",
            'metadata': {'confidence': 0.85, 'source_type': 'tutorial'}
        },
        {
            'kind': 'article_content',
            'title': 'Artificial Intelligence Overview',
            'text': """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to 
            human intelligence. AI is used in various applications including natural language processing, 
            computer vision, and expert systems. Machine learning is a subset of AI that enables computers 
            to learn without being explicitly programmed. Deep learning, in turn, is a subset of machine 
            learning that uses neural networks with multiple layers.""",
            'metadata': {'confidence': 0.8, 'source_type': 'article'}
        },
        {
            'kind': 'recipe_content',
            'title': 'Simple Pasta Recipe',
            'text': """How to make spaghetti aglio e olio:
            1. Boil salted water in a large pot
            2. Cook spaghetti according to package directions (8-10 minutes)
            3. Heat olive oil in a large pan over medium heat
            4. Add minced garlic and red pepper flakes, cook for 1 minute
            5. Reserve 1 cup pasta water, then drain pasta
            6. Add pasta to the pan with garlic oil
            7. Toss with pasta water, parsley, and parmesan cheese
            8. Serve immediately while hot""",
            'metadata': {'confidence': 0.9, 'source_type': 'recipe'}
        },
        {
            'kind': 'research_content',
            'title': 'Climate Change Information',
            'text': """Climate change refers to long-term changes in global temperatures and weather patterns. 
            While climate change is natural, human activities have been the main driver of climate change 
            since the 1800s, primarily through burning fossil fuels like coal, oil, and gas. The effects 
            include rising temperatures, melting ice caps, rising sea levels, and more extreme weather events.""",
            'metadata': {'confidence': 0.85, 'source_type': 'research'}
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
            "What is climate change?",
            "How do you make spaghetti aglio e olio?",
            "Compare artificial intelligence and machine learning",
            "List the steps to install Python",
            "What caused climate change?"
        ]
        
        print(f"\nTesting {len(demo_questions)} different types of questions...")
        
        # Test different types of questions
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{'='*20} Question {i} {'='*20}")
            print(f"Q: {question}")
            print("-" * 60)
            
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
                for j, source in enumerate(answer.sources[:3], 1):
                    title = source.title or f"{source.kind} #{source.artifact_id}"
                    print(f"   {j}. {title}")
                    if source.url:
                        print(f"      URL: {source.url}")
            
            # Show follow-up suggestions
            if answer.follow_up_suggestions:
                print(f"\n💡 Related Questions:")
                for suggestion in answer.follow_up_suggestions:
                    print(f"   • {suggestion}")
            
            if i < len(demo_questions):
                input("\nPress Enter to continue to next question...")
        
        # Demo session management
        print(f"\n{'='*20} Session Demo {'='*20}")
        print("Creating a QA session for conversational context...")
        
        session_id = qa_system.create_session(user_id="demo_user")
        print(f"✓ Session created: {session_id}")
        
        # Ask initial question
        print(f"\nQ1: What is Python used for?")
        answer1 = qa_system.ask_question("What is Python used for?", session_id)
        print(f"A1: {answer1.answer_text[:100]}...")
        
        # Ask follow-up question
        print(f"\nQ2: How does it compare to other languages?")
        answer2 = qa_system.ask_follow_up(
            "What is Python used for?",
            "How does it compare to other languages?",
            session_id
        )
        print(f"A2: {answer2.answer_text[:100]}...")
        
        # Show session statistics
        stats = qa_system.get_session_stats(session_id)
        print(f"\n📈 Session Statistics:")
        print(f"   • Questions asked: {stats['question_count']}")
        print(f"   • Average confidence: {stats['average_confidence']:.2f}")
        print(f"   • Topics covered: {len(stats['topics_covered'])}")
        
        qa_system.close_session(session_id)
        print(f"✓ Session closed")
        
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