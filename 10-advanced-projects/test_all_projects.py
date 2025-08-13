#!/usr/bin/env python3
"""
Module 10 Projects Validation Runner

This script runs and validates all Module 10 advanced projects with live models
to ensure they are working correctly with proper DSPy configuration.

Author: DSPy Learning Framework
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent / "08-custom-modules"))
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_multi_agent_system():
    """Test the multi-agent system with live models"""
    print("\n" + "=" * 60)
    print("🤖 Testing Multi-Agent System")
    print("=" * 60)

    try:
        from dspy_config import configure_dspy_lm, get_configured_model_info
        from multi_agent_system import AgentRole, MultiAgentResearchSystem

        # Configure DSPy
        success = configure_dspy_lm("auto")
        if not success:
            print("❌ Failed to configure DSPy with live models")
            return {"success": False, "error": "DSPy configuration failed"}

        model_info = get_configured_model_info()
        print(f"✅ Using model: {model_info.get('model', 'unknown')}")

        # Initialize system
        system = MultiAgentResearchSystem()
        await system.initialize_system()
        await system.start_system()

        print(f"✅ Created system with {len(system.agents)} agents")

        # Test simple task
        task = "Analyze the current trends in artificial intelligence research"
        print(f"🎯 Testing task: {task}")

        research_id = await system.conduct_research(
            topic=task, objectives=["identify key trends", "analyze impact"]
        )

        # Get results
        result = await system.get_research_results(research_id)

        # Stop system
        await system.stop_system()

        if result and result.get("status") == "completed":
            print(f"✅ Task completed successfully")
            print(f"   Research ID: {research_id}")
            print(f"   Result preview: {str(result)[:200]}...")
            return {"success": True, "result": result}
        else:
            print(
                f"❌ Task failed: {result.get('error', 'Unknown error') if result else 'No result'}"
            )
            return {
                "success": False,
                "error": result.get("error") if result else "No result",
            }

    except Exception as e:
        logger.error(f"Multi-agent system test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_research_system():
    """Test the integrated research system with live models"""
    print("\n" + "=" * 60)
    print("🔬 Testing Integrated Research System")
    print("=" * 60)

    try:
        from dspy_config import configure_dspy_lm, get_configured_model_info
        from integrated_research_system import IntegratedResearchPlatform

        # Configure DSPy
        success = configure_dspy_lm("auto")
        if not success:
            print("❌ Failed to configure DSPy with live models")
            return {"success": False, "error": "DSPy configuration failed"}

        model_info = get_configured_model_info()
        print(f"✅ Using model: {model_info.get('model', 'unknown')}")

        # Initialize research system
        system = IntegratedResearchPlatform()
        await system.initialize_platform()
        await system.start_platform()

        query_text = "What are the latest developments in quantum computing?"
        print(f"🔍 Testing query: {query_text}")

        # Create and start research project
        project_id = await system.create_research_project(
            project_name="Quantum Computing Research",
            description="Research on latest quantum computing developments",
            objectives=[query_text],
            template_name="comprehensive_research",
        )

        success = await system.start_research_project(project_id)

        if success:
            # Get project status
            result = await system.get_project_status(project_id)
        else:
            result = {"success": False, "error": "Failed to start project"}

        await system.stop_platform()

        if result.get("success"):
            print(f"✅ Research completed successfully")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"   Sources: {result.get('sources_count', 0)}")

            if "synthesis" in result:
                print(f"   Synthesis preview: {str(result['synthesis'])[:200]}...")

            return {"success": True, "result": result}
        else:
            print(f"❌ Research failed: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error")}

    except Exception as e:
        logger.error(f"Research system test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_document_processing():
    """Test the document processing system with live models"""
    print("\n" + "=" * 60)
    print("📄 Testing Document Processing System")
    print("=" * 60)

    try:
        from document_processing_system import DocumentProcessor, ExtractionType
        from dspy_config import configure_dspy_lm, get_configured_model_info

        # Configure DSPy
        success = configure_dspy_lm("auto")
        if not success:
            print("❌ Failed to configure DSPy with live models")
            return {"success": False, "error": "DSPy configuration failed"}

        model_info = get_configured_model_info()
        print(f"✅ Using model: {model_info.get('model', 'unknown')}")

        # Initialize processor
        processor = DocumentProcessor()

        # Create test document
        test_content = """
        Artificial Intelligence in Healthcare
        
        Artificial intelligence is transforming healthcare through various applications
        including medical imaging, drug discovery, and personalized treatment plans.
        Machine learning algorithms can analyze medical data to identify patterns
        and make predictions that assist healthcare professionals in diagnosis
        and treatment decisions.
        
        Key benefits include improved accuracy, faster processing, and the ability
        to handle large volumes of data. However, challenges remain in areas such
        as data privacy, regulatory compliance, and ensuring AI systems are
        transparent and explainable.
        """

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        print(f"📋 Processing test document...")

        # Process document
        result = await processor.process_document(
            file_path=temp_file,
            extraction_types=[
                ExtractionType.SUMMARY,
                ExtractionType.KEYWORDS,
                ExtractionType.TOPICS,
                ExtractionType.ENTITIES,
            ],
        )

        # Clean up
        Path(temp_file).unlink()

        if result.status.value == "completed":
            print(f"✅ Document processed successfully")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Quality score: {result.quality_score:.2f}")
            print(f"   Keywords: {', '.join(result.keywords[:5])}")
            print(f"   Topics: {', '.join(result.topics[:3])}")
            print(f"   Summary: {result.summary[:150]}...")

            return {"success": True, "result": result}
        else:
            print(f"❌ Document processing failed: {result.errors}")
            return {"success": False, "error": result.errors}

    except Exception as e:
        logger.error(f"Document processing test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_code_analysis():
    """Test the code analysis tool with live models"""
    print("\n" + "=" * 60)
    print("💻 Testing Code Analysis Tool")
    print("=" * 60)

    try:
        from code_analysis_tool import CodeAnalysisEngine, CodeLanguage
        from dspy_config import configure_dspy_lm, get_configured_model_info

        # Configure DSPy
        success = configure_dspy_lm("auto")
        if not success:
            print("❌ Failed to configure DSPy with live models")
            return {"success": False, "error": "DSPy configuration failed"}

        model_info = get_configured_model_info()
        print(f"✅ Using model: {model_info.get('model', 'unknown')}")

        # Initialize engine
        engine = CodeAnalysisEngine()

        # Test code
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"Fibonacci({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
        """

        print(f"🔍 Analyzing Python code...")

        # Analyze code
        result = await engine.analyze_code(test_code, "test.py")

        if result.quality_score > 0:
            print(f"✅ Code analysis completed")
            print(f"   Language: {result.language.value}")
            print(f"   Quality score: {result.quality_score:.2f}")
            print(f"   Lines of code: {result.metrics.lines_of_code}")
            print(f"   Complexity: {result.metrics.cyclomatic_complexity:.1f}")
            print(f"   Issues found: {len(result.issues)}")

            if result.summary:
                print(f"   Summary: {result.summary[:150]}...")

            return {"success": True, "result": result}
        else:
            print(f"❌ Code analysis failed: {result.errors}")
            return {"success": False, "error": result.errors}

    except Exception as e:
        logger.error(f"Code analysis test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_conversational_ai():
    """Test the conversational AI platform with live models"""
    print("\n" + "=" * 60)
    print("💬 Testing Conversational AI Platform")
    print("=" * 60)

    try:
        from conversational_ai_platform import ConversationalAIPlatform
        from dspy_config import configure_dspy_lm, get_configured_model_info

        # Configure DSPy
        success = configure_dspy_lm("auto")
        if not success:
            print("❌ Failed to configure DSPy with live models")
            return {"success": False, "error": "DSPy configuration failed"}

        model_info = get_configured_model_info()
        print(f"✅ Using model: {model_info.get('model', 'unknown')}")

        # Initialize platform
        platform = ConversationalAIPlatform(":memory:")

        # Start conversation
        conv_result = await platform.start_conversation(
            user_id="test_user", mode="casual", title="Test Conversation"
        )

        if not conv_result.get("success"):
            print(f"❌ Failed to start conversation: {conv_result.get('error')}")
            return {"success": False, "error": conv_result.get("error")}

        conversation_id = conv_result["conversation_id"]
        print(f"✅ Started conversation: {conversation_id}")

        # Test conversation
        test_messages = [
            "Hello! How are you today?",
            "Can you tell me about artificial intelligence?",
            "Thank you for the information!",
        ]

        conversation_results = []
        for i, message in enumerate(test_messages, 1):
            print(f"💬 Message {i}: {message}")

            response = await platform.send_message(conversation_id, message)
            conversation_results.append(response)

            if response.get("success"):
                print(f"🤖 Response: {response['response'][:100]}...")
                print(f"   Confidence: {response.get('confidence', 0):.2f}")
                print(f"   Processing time: {response.get('processing_time', 0):.2f}s")
            else:
                print(f"❌ Response failed: {response.get('error')}")

        # End conversation
        end_result = await platform.end_conversation(conversation_id)

        successful_messages = len([r for r in conversation_results if r.get("success")])

        if successful_messages > 0:
            print(f"✅ Conversation test completed")
            print(f"   Successful messages: {successful_messages}/{len(test_messages)}")

            if end_result.get("success"):
                analytics = end_result.get("final_analytics", {})
                print(f"   Final quality: {analytics.get('quality_score', 0):.2f}")

            return {"success": True, "successful_messages": successful_messages}
        else:
            print(f"❌ All conversation messages failed")
            return {"success": False, "error": "All messages failed"}

    except Exception as e:
        logger.error(f"Conversational AI test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


async def run_all_tests():
    """Run all Module 10 project tests"""
    print("🚀 Module 10 Advanced Projects - Live Model Validation")
    print("=" * 80)

    # Test functions
    tests = [
        ("Multi-Agent System", test_multi_agent_system),
        ("Research System", test_research_system),
        ("Document Processing", test_document_processing),
        ("Code Analysis", test_code_analysis),
        ("Conversational AI", test_conversational_ai),
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} test...")

        try:
            result = await test_func()
            results.append(
                {
                    "test": test_name,
                    "success": result.get("success", False),
                    "error": result.get("error") if not result.get("success") else None,
                }
            )
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append({"test": test_name, "success": False, "error": str(e)})

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    successful_tests = len([r for r in results if r["success"]])
    total_tests = len(results)

    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests / total_tests) * 100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")

    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {result['test']}: {status}")
        if result["error"]:
            print(f"    Error: {result['error']}")

    # Check DSPy configuration
    try:
        from dspy_config import get_configured_model_info

        model_info = get_configured_model_info()
        print(f"\n🔧 DSPy Configuration:")
        print(f"   Model: {model_info.get('model', 'unknown')}")
        print(f"   Provider: {model_info.get('provider', 'unknown')}")
        print(f"   Configured: {model_info.get('configured', False)}")
    except Exception as e:
        print(f"\n❌ Could not get DSPy configuration: {e}")

    if successful_tests == total_tests:
        print(f"\n🎉 All Module 10 projects validated successfully with live models!")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} projects need attention")

    return {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / total_tests,
        "results": results,
    }


if __name__ == "__main__":
    asyncio.run(run_all_tests())
