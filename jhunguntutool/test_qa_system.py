"""
Test suite for the ContextBox Question-Answering System

This test suite demonstrates and validates the QA system functionality including:
- Question classification
- Content retrieval and semantic search
- Answer generation with confidence scoring
- Source attribution and citation
- Session management
- Multi-document question answering
- Follow-up question handling
"""

import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the contextbox module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from contextbox.database import ContextDatabase, DatabaseError
from contextbox.llm import (
    QASystem,
    QuestionClassifier,
    ContentRetriever,
    AnswerGenerator,
    QuestionType,
    ConfidenceLevel,
    Source,
    QAContext
)


class QASystemTester:
    """Test harness for the QA system functionality."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_contextbox.db')
        self.database = ContextDatabase({'db_path': self.db_path})
        self.qa_system = QASystem(self.database)
        self.test_data = self._create_test_data()
        
    def _create_test_data(self) -> Dict[str, Any]:
        """Create test data for the database."""
        return {
            'wikipedia_content': {
                'Artificial Intelligence': """Artificial Intelligence (AI) is intelligence demonstrated by machines, 
                in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks 
                define the field as the study of "intelligent agents": any device that perceives its environment 
                and takes actions that maximize its chance of successfully achieving its goals. Colloquially, 
                the term "artificial intelligence" is often used to describe machines that mimic "cognitive" 
                functions that humans associate with the human mind, such as "learning" and "problem solving".""",
                
                'Machine Learning': """Machine learning (ML) is a field of inquiry devoted to understanding 
                and building methods that 'learn' – that is, methods that leverage data to improve performance 
                on some set of tasks. It is seen as a part of artificial intelligence. Machine learning 
                algorithms build a model based on training data in order to make predictions or decisions 
                without being explicitly programmed to do so.""",
                
                'Python Programming': """Python is an interpreted, high-level, general-purpose programming language. 
                Python's design philosophy emphasizes code readability with its notable use of significant indentation. 
                Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, 
                including structured, object-oriented, and functional programming. Python is often described as 
                a "batteries included" language due to its comprehensive standard library."""
            },
            
            'webpage_content': {
                'Tech Tutorial': """To install Python on Ubuntu, follow these steps:
                1. Update your package index: sudo apt update
                2. Install Python 3: sudo apt install python3
                3. Install pip: sudo apt install python3-pip
                4. Verify installation: python3 --version
                Python is widely used for web development, data science, automation, and artificial intelligence.""",
                
                'AI Research': """Recent advances in large language models have revolutionized natural language 
                processing. Models like GPT-3 and GPT-4 demonstrate remarkable capabilities in text generation, 
                translation, and reasoning. The transformer architecture, introduced in 2017, has become the 
                foundation for most modern AI systems. Current research focuses on improving efficiency, 
                reducing bias, and enhancing factual accuracy."""
            },
            
            'procedural_content': {
                'Cooking Recipe': """To make pasta al pomodoro:
                1. Boil water with salt
                2. Cook pasta according to package instructions
                3. Heat olive oil in a pan
                4. Add minced garlic and cook for 1 minute
                5. Add crushed tomatoes and simmer for 10 minutes
                6. Toss pasta with sauce and add fresh basil
                7. Serve with grated parmesan cheese""",
                
                'Git Workflow': """Git workflow for feature development:
                1. Create a new branch: git checkout -b feature-name
                2. Make your changes and commit them
                3. Push to remote: git push origin feature-name
                4. Create a pull request on GitHub
                5. Review and merge after approval"""
            }
        }
    
    def setup_test_database(self) -> None:
        """Populate the test database with sample content."""
        print("Setting up test database...")
        
        try:
            # Create a test capture
            capture_id = self.database.create_capture(
                source_window="Test Capture",
                notes="Test data for QA system"
            )
            
            # Add Wikipedia-style content artifacts
            for title, content in self.test_data['wikipedia_content'].items():
                self.database.create_artifact(
                    capture_id=capture_id,
                    kind='wikipedia_content',
                    title=title,
                    text=content,
                    metadata={
                        'extraction_method': 'test_data',
                        'confidence': 0.9,
                        'extraction_id': f'wiki_{title.lower().replace(" ", "_")}'
                    }
                )
            
            # Add webpage content artifacts
            for title, content in self.test_data['webpage_content'].items():
                self.database.create_artifact(
                    capture_id=capture_id,
                    kind='webpage_content',
                    title=title,
                    text=content,
                    metadata={
                        'extraction_method': 'test_data',
                        'confidence': 0.8,
                        'url': f'https://example.com/{title.lower().replace(" ", "_")}'
                    }
                )
            
            # Add procedural content artifacts
            for title, content in self.test_data['procedural_content'].items():
                self.database.create_artifact(
                    capture_id=capture_id,
                    kind='procedural_content',
                    title=title,
                    text=content,
                    metadata={
                        'extraction_method': 'test_data',
                        'confidence': 0.85,
                        'content_type': 'instructions'
                    }
                )
            
            print(f"✓ Test database setup complete with capture ID: {capture_id}")
            
        except Exception as e:
            print(f"✗ Error setting up test database: {e}")
            raise
    
    def test_question_classification(self) -> None:
        """Test question classification functionality."""
        print("\n=== Testing Question Classification ===")
        
        test_cases = [
            ("What is artificial intelligence?", QuestionType.FACTUAL),
            ("How do I install Python on Ubuntu?", QuestionType.PROCEDURAL),
            ("What are the differences between AI and ML?", QuestionType.COMPARATIVE),
            ("Why is Python popular for data science?", QuestionType.EXPLANATORY),
            ("When was machine learning invented?", QuestionType.FACTUAL),
            ("How does transformer architecture work?", QuestionType.INFERENTIAL),
            ("What are the steps to cook pasta?", QuestionType.PROCEDURAL),
            ("List the main features of Python", QuestionType.LIST),
            ("What caused the AI revolution?", QuestionType.CAUSAL),
            ("Something unclear and random", QuestionType.UNCLEAR)
        ]
        
        classifier = QuestionClassifier()
        
        for question, expected_type in test_cases:
            classified_type, confidence = classifier.classify(question)
            status = "✓" if classified_type == expected_type else "✗"
            print(f"{status} '{question[:40]}...' → {classified_type.value} (confidence: {confidence:.2f})")
    
    def test_content_retrieval(self) -> None:
        """Test content retrieval and semantic search."""
        print("\n=== Testing Content Retrieval ===")
        
        test_queries = [
            "artificial intelligence definition",
            "Python installation steps", 
            "machine learning vs AI",
            "pasta cooking instructions",
            "transformer architecture"
        ]
        
        retriever = ContentRetriever(self.database)
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            
            # Test keyword-based search
            keywords = retriever._extract_keywords(query)
            print(f"  Keywords: {keywords}")
            
            # Retrieve relevant content
            sources = retriever.retrieve_relevant_content(
                query, 
                QuestionType.FACTUAL,
                max_results=5
            )
            
            print(f"  Found {len(sources)} sources:")
            for i, source in enumerate(sources[:3], 1):
                relevance = source.relevance_score
                title = source.title or f"{source.kind} #{source.artifact_id}"
                print(f"    {i}. {title} (relevance: {relevance:.2f})")
    
    def test_answer_generation(self) -> None:
        """Test answer generation for different question types."""
        print("\n=== Testing Answer Generation ===")
        
        test_cases = [
            {
                'question': 'What is artificial intelligence?',
                'question_type': QuestionType.FACTUAL,
                'expected_keywords': ['artificial', 'intelligence', 'machines']
            },
            {
                'question': 'How do I install Python on Ubuntu?',
                'question_type': QuestionType.PROCEDURAL,
                'expected_keywords': ['install', 'python', 'ubuntu', 'steps']
            },
            {
                'question': 'What are the differences between AI and machine learning?',
                'question_type': QuestionType.COMPARATIVE,
                'expected_keywords': ['difference', 'compare', 'artificial', 'intelligence']
            }
        ]
        
        generator = AnswerGenerator()
        
        for test_case in test_cases:
            print(f"\nGenerating answer for: '{test_case['question']}'")
            
            # Retrieve context
            retriever = ContentRetriever(self.database)
            sources = retriever.retrieve_relevant_content(
                test_case['question'],
                test_case['question_type'],
                max_results=5
            )
            
            # Create QA context
            qa_context = QAContext(
                question=test_case['question'],
                question_type=test_case['question_type'],
                keywords=retriever._extract_keywords(test_case['question']),
                context_window=sources
            )
            
            # Generate answer
            answer = generator.generate_answer(qa_context, sources)
            
            print(f"  Answer type: {answer.question_type.value}")
            print(f"  Confidence: {answer.confidence_level.value} ({answer.confidence_score:.2f})")
            print(f"  Sources used: {len(answer.sources)}")
            print(f"  Processing time: {answer.processing_time:.2f}s")
            print(f"  Answer preview: {answer.answer_text[:100]}...")
    
    def test_full_qa_workflow(self) -> None:
        """Test the complete QA workflow from question to formatted answer."""
        print("\n=== Testing Full QA Workflow ===")
        
        test_questions = [
            "What is Python programming used for?",
            "How do you cook pasta al pomodoro?",
            "What caused the AI revolution?",
            "Compare artificial intelligence and machine learning"
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Q: {question}")
            print('='*60)
            
            # Ask the question
            answer = self.qa_system.ask_question(question)
            
            # Display results
            print(f"Answer: {answer.answer_text}")
            print(f"\nMetadata:")
            print(f"  • Question Type: {answer.question_type.value}")
            print(f"  • Confidence: {answer.confidence_level.value} ({answer.confidence_score:.2f})")
            print(f"  • Processing Time: {answer.processing_time:.2f}s")
            print(f"  • Sources: {len(answer.sources)}")
            
            if answer.sources:
                print(f"\nTop Sources:")
                for i, source in enumerate(answer.sources[:3], 1):
                    title = source.title or f"{source.kind} #{source.artifact_id}"
                    print(f"  {i}. {title}")
                    if source.url:
                        print(f"     URL: {source.url}")
            
            if answer.follow_up_suggestions:
                print(f"\nFollow-up Suggestions:")
                for suggestion in answer.follow_up_suggestions:
                    print(f"  • {suggestion}")
            
            if answer.explanations:
                print(f"\nExplanations:")
                for explanation in answer.explanations:
                    print(f"  • {explanation}")
    
    def test_session_management(self) -> None:
        """Test session management and follow-up questions."""
        print("\n=== Testing Session Management ===")
        
        # Create a session
        session_id = self.qa_system.create_session(user_id="test_user")
        print(f"Created session: {session_id}")
        
        # Ask initial question
        print("\nInitial question:")
        initial_answer = self.qa_system.ask_question("What is artificial intelligence?", session_id)
        print(f"Answer: {initial_answer.answer_text[:100]}...")
        
        # Ask follow-up question
        print("\nFollow-up question:")
        follow_up_answer = self.qa_system.ask_follow_up(
            "What is artificial intelligence?",
            "How is it different from machine learning?",
            session_id
        )
        print(f"Answer: {follow_up_answer.answer_text[:100]}...")
        
        # Get session statistics
        stats = self.qa_system.get_session_stats(session_id)
        print(f"\nSession Statistics:")
        print(f"  • Questions asked: {stats['question_count']}")
        print(f"  • Average confidence: {stats['average_confidence']:.2f}")
        print(f"  • Total processing time: {stats['total_processing_time']:.2f}s")
        
        # Close session
        self.qa_system.close_session(session_id)
        print(f"Closed session: {session_id}")
    
    def test_content_search(self) -> None:
        """Test direct content search functionality."""
        print("\n=== Testing Content Search ===")
        
        search_queries = [
            "Python programming",
            "artificial intelligence",
            "installation steps",
            "machine learning"
        ]
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            results = self.qa_system.search_content(query, limit=5)
            
            print(f"  Found {len(results)} results:")
            for result in results[:3]:
                title = result.get('title', f"{result['kind']} #{result['id']}")
                content_preview = result.get('text', '')[:100] + "..." if result.get('text') else ""
                print(f"    • {title}")
                if content_preview:
                    print(f"      {content_preview}")
    
    def test_performance_optimization(self) -> None:
        """Test performance and optimization features."""
        print("\n=== Testing Performance Optimization ===")
        
        # Test query optimization
        print("Analyzing query patterns...")
        optimization_recommendations = self.qa_system.optimize_queries()
        
        print("Optimization Recommendations:")
        for category, items in optimization_recommendations.items():
            print(f"  {category}:")
            for item in items:
                print(f"    • {item}")
        
        # Test system statistics
        print("\nSystem Statistics:")
        system_stats = self.qa_system.get_qa_statistics()
        
        print(f"  Performance Metrics:")
        perf_metrics = system_stats['performance_metrics']
        print(f"    • Success Rate: {perf_metrics['success_rate']:.2%}")
        print(f"    • Average Processing Time: {perf_metrics['average_processing_time']:.2f}s")
        print(f"    • Questions per Minute: {perf_metrics['questions_per_minute']:.1f}")
        
        print(f"  Database Stats:")
        db_stats = system_stats['database_stats']
        print(f"    • Total Captures: {db_stats['total_captures']}")
        print(f"    • Total Artifacts: {db_stats['total_artifacts']}")
    
    def test_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")
        
        edge_cases = [
            ("", "Empty question"),
            ("?", "Just a question mark"),
            ("a", "Very short question"),
            ("This is a very long question " * 10, "Very long question"),
            ("What is the answer to life, the universe, and everything?", "Impossible question"),
            ("How do I do something that doesn't exist?", "Non-existent procedure"),
            ("Compare X to Y when I have no content about either", "No relevant content")
        ]
        
        for question, description in edge_cases:
            print(f"\nTesting: {description}")
            print(f"Question: '{question[:50]}...'")
            
            try:
                answer = self.qa_system.ask_question(question)
                print(f"  Response: {answer.answer_text[:100]}...")
                print(f"  Confidence: {answer.confidence_level.value}")
                print(f"  Needs clarification: {answer.needs_clarification}")
                
                if answer.needs_clarification and answer.clarification_questions:
                    print(f"  Clarifications:")
                    for clarification in answer.clarification_questions:
                        print(f"    • {clarification}")
                        
            except Exception as e:
                print(f"  Error: {e}")
    
    def run_comprehensive_test(self) -> None:
        """Run all tests in sequence."""
        print("="*80)
        print("CONTEXTBOX QUESTION-ANSWERING SYSTEM TEST SUITE")
        print("="*80)
        
        try:
            # Setup
            self.setup_test_database()
            
            # Run tests
            self.test_question_classification()
            self.test_content_retrieval()
            self.test_answer_generation()
            self.test_full_qa_workflow()
            self.test_session_management()
            self.test_content_search()
            self.test_performance_optimization()
            self.test_edge_cases()
            
            print("\n" + "="*80)
            print("TEST SUITE COMPLETED SUCCESSFULLY")
            print("="*80)
            
        except Exception as e:
            print(f"\n✗ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up test resources."""
        try:
            self.database.cleanup()
            # Clean up temp directory
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"\n✓ Cleaned up test resources")
        except Exception as e:
            print(f"\n⚠ Error during cleanup: {e}")


def main():
    """Main function to run the test suite."""
    tester = QASystemTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()