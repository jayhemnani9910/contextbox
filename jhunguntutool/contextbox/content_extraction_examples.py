"""
Content Extraction Integration Examples

This file demonstrates how to use the integrated content extraction capabilities
in ContextBox, including automatic extraction during capture, manual extraction
via API and CLI, and configuration options.

Examples cover:
1. Automatic content extraction during context capture
2. Manual content extraction via API
3. CLI usage examples
4. Configuration options
5. Error handling and best practices
6. Database integration and queries
"""

import json
import os
from pathlib import Path
from contextbox import ContextBox

# =============================================================================
# 1. AUTOMATIC CONTENT EXTRACTION DURING CAPTURE
# =============================================================================

def example_automatic_extraction():
    """Example of automatic content extraction when URLs are found during capture."""
    print("=== Automatic Content Extraction Example ===")
    
    # Initialize ContextBox with content extraction enabled
    config = {
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True,  # Automatically extract content when URLs found
            'store_in_database': True,
            'enabled_extractors': ['text_extraction', 'system_extraction']
        },
        'database': {
            'db_path': 'contextbox.db'
        }
    }
    
    contextbox = ContextBox(config)
    
    # Simulate context capture data with URLs
    capture_data = {
        'timestamp': '2023-01-01T12:00:00',
        'active_window': {
            'title': 'Visit https://python.org for Python documentation',
            'application': 'Web Browser'
        },
        'text': 'Check out https://github.com/python/cpython and https://docs.python.org',
        'screenshot_path': None,  # Would be path to screenshot
        'clipboard_text': 'Interesting article at https://realpython.com'
    }
    
    # Extract content automatically (called when URLs are found)
    extraction_result = contextbox.extract_content_from_capture(capture_data)
    
    print(f"Extraction ID: {extraction_result['extraction_id']}")
    print(f"Total URLs found: {extraction_result['metadata']['total_urls_found']}")
    print(f"Successful extractions: {extraction_result['metadata']['successful_extractions']}")
    
    if 'url_analysis' in extraction_result:
        analysis = extraction_result['url_analysis']
        print(f"URL types: {analysis.get('by_type', {})}")
        print(f"High confidence URLs: {len(analysis.get('high_confidence', []))}")
    
    # Results are automatically stored in database
    print("Results automatically stored in database!")


# =============================================================================
# 2. MANUAL CONTENT EXTRACTION VIA API
# =============================================================================

def example_manual_extraction_api():
    """Example of manual content extraction via API."""
    print("\n=== Manual Content Extraction via API ===")
    
    contextbox = ContextBox({
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': False,  # Manual mode
            'output_format': 'json'
        }
    })
    
    # Input data for extraction
    input_data = {
        'text': '''
        Visit these sites for more information:
        - https://docs.python.org (Python documentation)
        - https://github.com/psf/requests (Requests library)
        - https://stackoverflow.com (Programming Q&A)
        
        Also check out python.org and github.com for more resources.
        ''',
        'window_title': 'Python Development Resources',
        'context': 'Documentation and learning resources'
    }
    
    # Perform manual extraction
    result = contextbox.extract_content_manually(
        input_data=input_data,
        extract_urls=True,
        extract_text=True,
        extract_images=False,
        output_format='pretty'
    )
    
    print(result)  # Pretty formatted output
    
    # Get configuration
    config = contextbox.get_content_extraction_config()
    print(f"\nExtraction features enabled: {config['features']}")


# =============================================================================
# 3. CLI USAGE EXAMPLES
# =============================================================================

def example_cli_usage():
    """Example of CLI usage for content extraction."""
    print("\n=== CLI Usage Examples ===")
    
    # Example 1: Extract content from text file
    print("1. Extract content from text file:")
    print("   contextbox extract-content input.txt --format pretty")
    
    # Example 2: Extract from JSON file with custom configuration
    print("\n2. Extract from JSON file with config:")
    print("   contextbox extract-content data.json --type json --output results.json")
    
    # Example 3: Extract only URLs from image (with OCR)
    print("\n3. Extract URLs from image:")
    print("   contextbox extract-content screenshot.png --type image --extract-text --extract-urls")
    
    # Example 4: Batch extraction with specific format
    print("\n4. Extract with detailed output:")
    print("   contextbox extract-content input.txt --format detailed --output detailed_results.json")
    
    # Show CLI help
    print("\n5. Get help:")
    print("   contextbox extract-content --help")


def create_sample_input_files():
    """Create sample input files for CLI examples."""
    print("\n=== Creating Sample Input Files ===")
    
    # Create sample text file
    text_file = "sample_text.txt"
    with open(text_file, 'w') as f:
        f.write("""
        Useful Programming Resources:
        
        Documentation:
        - Python: https://docs.python.org
        - JavaScript: https://developer.mozilla.org
        - Git: https://git-scm.com/docs
        
        Learning Platforms:
        - Codecademy: https://codecademy.com
        - FreeCodeCamp: https://freecodecamp.org
        - Coursera: https://coursera.org
        
        Open Source Projects:
        - GitHub: https://github.com
        - GitLab: https://gitlab.com
        
        Email contact: developer@example.com
        """)
    
    # Create sample JSON file
    json_file = "sample_data.json"
    sample_data = {
        "project_info": {
            "name": "ContextBox Project",
            "repository": "https://github.com/example/contextbox",
            "documentation": "https://contextbox.readthedocs.io"
        },
        "links": [
            "https://python.org",
            "https://stackoverflow.com",
            "https://realpython.com"
        ],
        "description": "A tool for capturing and extracting digital context"
    }
    
    with open(json_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created {text_file} and {json_file}")


# =============================================================================
# 4. CONFIGURATION OPTIONS
# =============================================================================

def example_configuration():
    """Example of different configuration options."""
    print("\n=== Configuration Examples ===")
    
    # Basic configuration
    basic_config = {
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True,
            'output_format': 'json',
            'store_in_database': True
        }
    }
    
    # Advanced configuration
    advanced_config = {
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True,
            'output_format': 'pretty',
            'store_in_database': True,
            'enabled_extractors': [
                'text_extraction',
                'system_extraction', 
                'network_extraction'
            ],
            'extractors': {
                'ocr': {
                    'enhance_contrast': True,
                    'sharpen': True,
                    'grayscale': True,
                    'min_image_size': 300,
                    'psm': 6,
                    'languages': 'eng'
                },
                'confidence_threshold': 0.7,
                'infer_domains': True
            }
        },
        'database': {
            'db_path': 'contextbox.db',
            'timeout': 30.0
        }
    }
    
    # Initialize with advanced config
    contextbox = ContextBox(advanced_config)
    
    # Update configuration at runtime
    config_updates = {
        'auto_extract': False,
        'output_format': 'summary'
    }
    contextbox.update_content_extraction_config(config_updates)
    
    print("Advanced configuration applied!")
    
    # Get current configuration
    current_config = contextbox.get_content_extraction_config()
    print(f"Current features: {current_config['features']}")


# =============================================================================
# 5. ERROR HANDLING AND BEST PRACTICES
# =============================================================================

def example_error_handling():
    """Example of error handling and best practices."""
    print("\n=== Error Handling Examples ===")
    
    contextbox = ContextBox({
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True
        }
    })
    
    try:
        # Example 1: Handle missing content extractor
        contextbox_no_extractor = ContextBox({
            'enable_content_extraction': False
        })
        
        # This will fall back to basic extraction
        result = contextbox_no_extractor.extract_content_from_capture({'text': 'test'})
        print("✅ Fallback to basic extraction works")
        
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
    
    try:
        # Example 2: Handle OCR availability
        if not contextbox.content_extractor.features['ocr']:
            print("⚠️  OCR not available, disabling image extraction")
        
        # Example 3: Validate input data
        input_data = {'text': ''}  # Empty text
        result = contextbox.extract_content_manually(
            input_data=input_data,
            extract_urls=True,
            extract_text=True,
            extract_images=False
        )
        print("✅ Handled empty input gracefully")
        
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
    
    try:
        # Example 4: Check feature availability
        config = contextbox.get_content_extraction_config()
        available_features = [k for k, v in config['features'].items() if v]
        print(f"✅ Available features: {available_features}")
        
    except Exception as e:
        print(f"❌ Feature check failed: {e}")


# =============================================================================
# 6. DATABASE INTEGRATION AND QUERIES
# =============================================================================

def example_database_integration():
    """Example of database integration and queries."""
    print("\n=== Database Integration Examples ===")
    
    contextbox = ContextBox({
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True,
            'store_in_database': True
        },
        'database': {
            'db_path': 'contextbox.db'
        }
    })
    
    # Simulate some extraction activity
    capture_data = {
        'active_window': {'title': 'Python Documentation'},
        'text': 'Visit https://python.org and https://docs.python.org'
    }
    
    result = contextbox.extract_content_from_capture(capture_data)
    capture_id = result.get('capture_id', 1)  # Assume capture ID
    
    try:
        # Get database statistics
        stats = contextbox.database.get_stats()
        print(f"Database stats: {stats}")
        
        # Get extraction results for a capture
        extraction_results = contextbox.database.get_extraction_results(capture_id)
        print(f"Found {len(extraction_results)} extraction artifacts")
        
        # Search for specific terms
        search_results = contextbox.database.search_extraction_artifacts('python')
        print(f"Found {len(search_results)} artifacts containing 'python'")
        
        # List recent captures
        captures = contextbox.database.list_captures(limit=5)
        print(f"Recent captures: {len(captures)}")
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")


# =============================================================================
# 7. BATCH PROCESSING EXAMPLE
# =============================================================================

def example_batch_processing():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    contextbox = ContextBox({
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True
        }
    })
    
    # Create sample files
    files_to_process = []
    
    # Create multiple sample files
    for i in range(3):
        filename = f"sample_file_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"""
            File {i} Content:
            
            Visit these resources:
            - https://example{i}.com
            - https://test{i}.org
            
            Contact: user{i}@example.com
            """)
        files_to_process.append(filename)
    
    try:
        results = []
        
        # Process each file
        for filename in files_to_process:
            print(f"Processing {filename}...")
            
            try:
                # Load file content
                with open(filename, 'r') as f:
                    content = f.read()
                
                # Extract content
                result = contextbox.extract_content_manually(
                    input_data={'text': content},
                    extract_urls=True,
                    extract_text=True,
                    extract_images=False,
                    output_format='json'
                )
                
                results.append({
                    'file': filename,
                    'success': True,
                    'urls_found': len(json.loads(result).get('extracted_content', {}).get('urls', {}).get('direct_urls', []))
                })
                
            except Exception as e:
                results.append({
                    'file': filename,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Processed {successful}/{len(files_to_process)} files successfully")
        
        for result in results:
            if result['success']:
                print(f"  {result['file']}: {result['urls_found']} URLs found")
            else:
                print(f"  {result['file']}: Failed - {result['error']}")
    
    finally:
        # Clean up files
        for filename in files_to_process:
            if os.path.exists(filename):
                os.remove(filename)


# =============================================================================
# 8. COMPREHENSIVE WORKFLOW EXAMPLE
# =============================================================================

def example_comprehensive_workflow():
    """Example of comprehensive content extraction workflow."""
    print("\n=== Comprehensive Workflow Example ===")
    
    # Initialize ContextBox with comprehensive configuration
    config = {
        'enable_content_extraction': True,
        'content_extraction': {
            'auto_extract': True,
            'output_format': 'json',
            'store_in_database': True,
            'enabled_extractors': ['text_extraction', 'system_extraction', 'network_extraction'],
            'extractors': {
                'ocr': {
                    'enhance_contrast': True,
                    'sharpen': True,
                    'psm': 6
                }
            }
        },
        'database': {
            'db_path': 'contextbox.db',
            'backup_interval': 3600  # 1 hour
        },
        'log_level': 'INFO'
    }
    
    contextbox = ContextBox(config)
    
    print("ContextBox initialized with comprehensive configuration")
    
    # Check available features
    features = contextbox.get_content_extraction_config()
    print(f"Available features: {[k for k, v in features['features'].items() if v]}")
    
    # Simulate a complete workflow
    workflow_steps = []
    
    try:
        # Step 1: Automatic extraction from capture
        print("\n1. Performing automatic extraction from capture...")
        capture_data = {
            'timestamp': '2023-01-01T12:00:00',
            'active_window': {
                'title': 'Development Environment - Python Project',
                'application': 'IDE'
            },
            'text': '''
            Working on the ContextBox project.
            
            Resources:
            - GitHub: https://github.com/contextbox/contextbox
            - Documentation: https://contextbox.readthedocs.io
            - Issues: https://github.com/contextbox/contextbox/issues
            
            Also checking:
            - Python docs: https://docs.python.org
            - Stack Overflow: https://stackoverflow.com
            ''',
            'screenshot_path': None
        }
        
        auto_result = contextbox.extract_content_from_capture(capture_data)
        workflow_steps.append({
            'step': 'automatic_extraction',
            'success': True,
            'urls_found': auto_result['metadata']['total_urls_found']
        })
        
        # Step 2: Manual extraction from specific data
        print("\n2. Performing manual extraction from curated data...")
        manual_input = {
            'project_links': [
                'https://python.org',
                'https://github.com/psf/requests',
                'https://docs.python.org/3/',
                'https://realpython.com'
            ],
            'description': 'Python development resources and documentation'
        }
        
        manual_result = contextbox.extract_content_manually(
            input_data=manual_input,
            extract_urls=True,
            extract_text=True,
            extract_images=False,
            output_format='json'
        )
        
        workflow_steps.append({
            'step': 'manual_extraction',
            'success': True,
            'urls_found': len(json.loads(manual_result).get('extracted_content', {}).get('urls', {}).get('direct_urls', []))
        })
        
        # Step 3: Database analysis
        print("\n3. Analyzing stored extraction results...")
        stats = contextbox.database.get_stats()
        workflow_steps.append({
            'step': 'database_analysis',
            'success': True,
            'stats': stats
        })
        
        # Step 4: Search and retrieve
        print("\n4. Searching extraction artifacts...")
        search_results = contextbox.database.search_extraction_artifacts('github', limit=10)
        workflow_steps.append({
            'step': 'artifact_search',
            'success': True,
            'results_found': len(search_results)
        })
        
        # Workflow summary
        print("\n=== Workflow Summary ===")
        for step in workflow_steps:
            status = "✅" if step['success'] else "❌"
            print(f"{status} {step['step']}: Success")
            if 'urls_found' in step:
                print(f"   URLs found: {step['urls_found']}")
            if 'stats' in step:
                print(f"   Database stats: {step['stats']}")
            if 'results_found' in step:
                print(f"   Search results: {step['results_found']}")
        
        print(f"\n✅ Comprehensive workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        for step in workflow_steps:
            if not step['success']:
                print(f"Failed at step: {step['step']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all examples."""
    print("ContextBox Content Extraction Integration Examples")
    print("=" * 60)
    
    # Create sample files for CLI examples
    create_sample_input_files()
    
    # Run examples
    try:
        example_automatic_extraction()
        example_manual_extraction_api()
        example_cli_usage()
        example_configuration()
        example_error_handling()
        example_database_integration()
        example_batch_processing()
        example_comprehensive_workflow()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try running: contextbox extract-content --help")
        print("2. Test with your own files")
        print("3. Explore the database queries")
        print("4. Customize the configuration")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
