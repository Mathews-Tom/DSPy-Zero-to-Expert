#!/usr/bin/env python3
"""
DSPy Learning Framework - Module 10: Advanced Projects Solutions

This script contains complete solutions for all exercises in Module 10: Advanced Projects.
Each solution demonstrates sophisticated DSPy applications including multi-agent systems,
research assistants, document processing, code analysis, and conversational AI.

Learning Objectives Covered:
- Building complex multi-agent systems with DSPy
- Creating intelligent research and analysis tools
- Implementing document processing pipelines
- Developing code analysis and generation systems
- Building conversational AI platforms
- Advanced optimization and evaluation techniques

Author: DSPy Learning Framework
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import dspy
from code_analysis_tool import CodeAnalysisEngine, CodeLanguage
from conversational_ai_platform import ConversationalAIPlatform, ConversationMode
from document_processing_system import DocumentProcessor, ExtractionType
from integrated_research_system import ResearchQuery, ResearchSystem

# Import the advanced project modules
from multi_agent_system import Agent, AgentRole, MultiAgentSystem
from utilities.dspy_helpers import (
    benchmark_signature_performance,
    setup_dspy_environment,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def exercise_10_1_multi_agent_system():
    """
    Exercise 10.1: Build Multi-Agent System

    This exercise demonstrates:
    - Creating and coordinating multiple AI agents
    - Agent communication and task delegation
    - Collaborative problem-solving workflows
    - System orchestration and management
    """
    print("\n" + "=" * 60)
    print("Exercise 10.1: Multi-Agent System")
    print("=" * 60)

    try:
        # Initialize the multi-agent system
        system = MultiAgentSystem()

        # Create specialized agents
        researcher = Agent(
            agent_id="researcher_001",
            name="Research Specialist",
            role=AgentRole.RESEARCHER,
            capabilities=["information_gathering", "analysis", "synthesis"],
            specialization="academic_research",
        )

        analyst = Agent(
            agent_id="analyst_001",
            name="Data Analyst",
            role=AgentRole.ANALYST,
            capabilities=["data_analysis", "pattern_recognition", "reporting"],
            specialization="quantitative_analysis",
        )

        coordinator = Agent(
            agent_id="coordinator_001",
            name="Task Coordinator",
            role=AgentRole.COORDINATOR,
            capabilities=[
                "task_management",
                "workflow_orchestration",
                "quality_control",
            ],
            specialization="project_management",
        )

        # Register agents
        system.register_agent(researcher)
        system.register_agent(analyst)
        system.register_agent(coordinator)

        print(f"✅ Created multi-agent system with {len(system.agents)} agents")

        # Test collaborative task
        task_description = """
        Research the current state of artificial intelligence in healthcare,
        analyze the key trends and challenges, and provide a comprehensive
        report with actionable insights.
        """

        print(f"\n🎯 Executing collaborative task...")
        print(f"Task: {task_description[:100]}...")

        # Execute the task
        result = await system.execute_collaborative_task(
            task_description=task_description,
            required_capabilities=[
                "information_gathering",
                "data_analysis",
                "reporting",
            ],
            max_execution_time=300,  # 5 minutes
        )

        if result["success"]:
            print(f"✅ Task completed successfully!")
            print(f"   Execution time: {result['execution_time']:.2f}s")
            print(f"   Agents involved: {len(result['participating_agents'])}")
            print(f"   Workflow steps: {len(result['workflow_steps'])}")
            print(f"   Result preview: {result['final_result'][:200]}...")
        else:
            print(f"❌ Task failed: {result['error']}")

        # Test agent communication
        print(f"\n💬 Testing agent communication...")
        comm_result = await system.facilitate_agent_communication(
            sender_id="researcher_001",
            receiver_id="analyst_001",
            message="I've gathered data on AI healthcare applications. Can you analyze the adoption trends?",
            message_type="data_request",
        )

        if comm_result["success"]:
            print(f"✅ Communication successful")
            print(f"   Response: {comm_result['response'][:150]}...")

        # Get system analytics
        analytics = system.get_system_analytics()
        print(f"\n📊 System Analytics:")
        print(f"   Total agents: {analytics['total_agents']}")
        print(f"   Active tasks: {analytics['active_tasks']}")
        print(f"   Completed tasks: {analytics['completed_tasks']}")
        print(f"   Success rate: {analytics['success_rate']:.1%}")

        return {
            "exercise": "10.1",
            "success": True,
            "agents_created": len(system.agents),
            "task_result": result,
            "analytics": analytics,
        }

    except Exception as e:
        logger.error(f"Exercise 10.1 failed: {e}")
        return {"exercise": "10.1", "success": False, "error": str(e)}


async def exercise_10_2_research_system():
    """
    Exercise 10.2: Integrated Research System

    This exercise demonstrates:
    - Automated research query processing
    - Multi-source information gathering
    - Intelligent synthesis and analysis
    - Research workflow automation
    """
    print("\n" + "=" * 60)
    print("Exercise 10.2: Integrated Research System")
    print("=" * 60)

    try:
        # Initialize research system
        research_system = ResearchSystem()

        # Create research query
        query = ResearchQuery(
            query_text="What are the latest developments in quantum computing for machine learning?",
            research_depth="comprehensive",
            domains=[
                "quantum_computing",
                "machine_learning",
                "artificial_intelligence",
            ],
            time_range="last_2_years",
            source_types=["academic_papers", "industry_reports", "news_articles"],
        )

        print(f"🔍 Executing research query...")
        print(f"Query: {query.query_text}")
        print(f"Domains: {', '.join(query.domains)}")

        # Execute research
        research_result = await research_system.execute_research(query)

        if research_result["success"]:
            print(f"✅ Research completed successfully!")
            print(f"   Sources found: {research_result['sources_count']}")
            print(f"   Processing time: {research_result['processing_time']:.2f}s")
            print(f"   Confidence score: {research_result['confidence_score']:.2f}")

            # Display key findings
            if "key_findings" in research_result:
                print(f"\n📋 Key Findings:")
                for i, finding in enumerate(research_result["key_findings"][:3], 1):
                    print(f"   {i}. {finding[:100]}...")

            # Display synthesis
            if "synthesis" in research_result:
                print(f"\n🔬 Research Synthesis:")
                print(f"   {research_result['synthesis'][:300]}...")
        else:
            print(f"❌ Research failed: {research_result['error']}")

        # Test research workflow automation
        print(f"\n⚙️ Testing workflow automation...")
        workflow_result = await research_system.create_research_workflow(
            workflow_name="AI_Healthcare_Analysis",
            research_steps=[
                "literature_review",
                "trend_analysis",
                "gap_identification",
                "recommendation_generation",
            ],
            automation_level="high",
        )

        if workflow_result["success"]:
            print(f"✅ Workflow created: {workflow_result['workflow_id']}")
            print(f"   Steps: {len(workflow_result['workflow_steps'])}")

        # Get research analytics
        analytics = research_system.get_research_analytics()
        print(f"\n📊 Research Analytics:")
        print(f"   Total queries processed: {analytics['total_queries']}")
        print(f"   Average processing time: {analytics['avg_processing_time']:.2f}s")
        print(f"   Success rate: {analytics['success_rate']:.1%}")

        return {
            "exercise": "10.2",
            "success": True,
            "research_result": research_result,
            "workflow_created": workflow_result.get("success", False),
            "analytics": analytics,
        }

    except Exception as e:
        logger.error(f"Exercise 10.2 failed: {e}")
        return {"exercise": "10.2", "success": False, "error": str(e)}


async def exercise_10_3_document_processing():
    """
    Exercise 10.3: Document Processing System

    This exercise demonstrates:
    - Multi-format document parsing and analysis
    - Intelligent information extraction
    - Document classification and routing
    - Batch processing capabilities
    """
    print("\n" + "=" * 60)
    print("Exercise 10.3: Document Processing System")
    print("=" * 60)

    try:
        # Initialize document processor
        processor = DocumentProcessor()

        # Create sample documents for testing
        sample_documents = {
            "research_paper.txt": """
            Abstract: This paper presents a novel approach to natural language processing
            using transformer architectures. We demonstrate significant improvements in
            performance across multiple benchmarks including GLUE and SuperGLUE.
            
            Introduction: Natural language processing has seen remarkable advances with
            the introduction of transformer models. However, challenges remain in
            efficiency and interpretability.
            
            Methodology: We propose a new attention mechanism that reduces computational
            complexity while maintaining performance. Our approach uses sparse attention
            patterns and dynamic routing.
            
            Results: Experimental results show 15% improvement in accuracy and 30%
            reduction in training time compared to baseline models.
            """,
            "business_report.txt": """
            Executive Summary: Q3 2024 financial results show strong growth across
            all business segments. Revenue increased 22% year-over-year to $2.1B.
            
            Key Metrics:
            - Revenue: $2.1B (+22% YoY)
            - Net Income: $340M (+18% YoY)
            - Customer Growth: 1.2M new customers
            - Market Share: 15.3% (+2.1% YoY)
            
            Outlook: We expect continued growth in Q4 with projected revenue of $2.3B.
            Key focus areas include product innovation and market expansion.
            """,
        }

        print(f"📄 Processing {len(sample_documents)} sample documents...")

        # Process documents
        processing_results = []
        for filename, content in sample_documents.items():
            # Create temporary file
            temp_file = Path(f"/tmp/{filename}")
            temp_file.write_text(content)

            # Process document
            result = await processor.process_document(
                file_path=temp_file,
                extraction_types=[
                    ExtractionType.SUMMARY,
                    ExtractionType.KEYWORDS,
                    ExtractionType.ENTITIES,
                    ExtractionType.TOPICS,
                ],
            )

            processing_results.append(result)

            print(f"\n📋 Results for {filename}:")
            print(f"   Status: {result.status.value}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Quality score: {result.quality_score:.2f}")
            print(f"   Keywords: {', '.join(result.keywords[:5])}")
            print(f"   Topics: {', '.join(result.topics[:3])}")
            print(f"   Summary: {result.summary[:150]}...")

            # Clean up
            temp_file.unlink()

        # Test batch processing
        print(f"\n⚡ Testing batch processing...")
        batch_files = [f"/tmp/batch_{i}.txt" for i in range(3)]
        for i, file_path in enumerate(batch_files):
            Path(file_path).write_text(
                f"Sample document {i+1} content for batch processing test."
            )

        batch_results = await processor.batch_process_documents(
            batch_files,
            extraction_types=[ExtractionType.SUMMARY, ExtractionType.KEYWORDS],
        )

        successful_batch = len(
            [r for r in batch_results if r.status.value == "completed"]
        )
        print(f"✅ Batch processing: {successful_batch}/{len(batch_files)} successful")

        # Clean up batch files
        for file_path in batch_files:
            Path(file_path).unlink(missing_ok=True)

        # Get processing analytics
        doc_list = processor.list_processed_documents()
        print(f"\n📊 Processing Analytics:")
        print(f"   Total documents processed: {len(doc_list)}")
        print(
            f"   Average quality score: {sum(d['quality_score'] for d in doc_list) / len(doc_list):.2f}"
        )

        return {
            "exercise": "10.3",
            "success": True,
            "documents_processed": len(processing_results),
            "batch_success_rate": successful_batch / len(batch_files),
            "average_quality": sum(r.quality_score for r in processing_results)
            / len(processing_results),
        }

    except Exception as e:
        logger.error(f"Exercise 10.3 failed: {e}")
        return {"exercise": "10.3", "success": False, "error": str(e)}


async def exercise_10_4_code_analysis():
    """
    Exercise 10.4: Code Analysis and Generation Tool

    This exercise demonstrates:
    - Multi-language code analysis
    - Quality assessment and improvement suggestions
    - Automated code generation
    - Refactoring recommendations
    """
    print("\n" + "=" * 60)
    print("Exercise 10.4: Code Analysis and Generation Tool")
    print("=" * 60)

    try:
        # Initialize code analysis engine
        engine = CodeAnalysisEngine()

        # Sample code for analysis
        sample_codes = {
            "python_function.py": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    for i in range(10):
        print(f"Fibonacci({i}) = {calculate_fibonacci(i)}")

if __name__ == "__main__":
    main()
            """,
            "javascript_class.js": """
class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        var result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return this.history;
    }
}

var calc = new Calculator();
console.log(calc.add(5, 3));
            """,
        }

        print(f"🔍 Analyzing {len(sample_codes)} code samples...")

        analysis_results = []
        for filename, code in sample_codes.items():
            # Analyze code
            result = await engine.analyze_code(code, filename)
            analysis_results.append(result)

            print(f"\n📋 Analysis for {filename}:")
            print(f"   Language: {result.language.value}")
            print(f"   Quality Score: {result.quality_score:.2f}")
            print(f"   Lines of Code: {result.metrics.lines_of_code}")
            print(f"   Complexity: {result.metrics.cyclomatic_complexity:.1f}")
            print(f"   Issues Found: {len(result.issues)}")

            # Show top issues
            if result.issues:
                print(f"   Top Issues:")
                for issue in result.issues[:2]:
                    print(f"     - {issue.level.value.upper()}: {issue.message}")

            # Show recommendations
            if result.recommendations:
                print(f"   Recommendations:")
                for rec in result.recommendations[:2]:
                    print(f"     - {rec}")

        # Test code generation
        print(f"\n🔧 Testing code generation...")
        generation_result = await engine.generate_code(
            requirements="Create a Python function that sorts a list using bubble sort algorithm",
            language=CodeLanguage.PYTHON,
        )

        if "error" not in generation_result:
            print(f"✅ Code generation successful")
            print(f"   Generated code preview:")
            print(
                "   "
                + "\n   ".join(
                    generation_result["generated_code"][:200].split("\n")[:5]
                )
            )
        else:
            print(f"❌ Code generation failed: {generation_result['error']}")

        # Test code explanation
        print(f"\n📖 Testing code explanation...")
        explanation_result = await engine.explain_code(
            sample_codes["python_function.py"], explanation_level="beginner"
        )

        if "error" not in explanation_result:
            print(f"✅ Code explanation successful")
            print(
                f"   Explanation preview: {explanation_result['explanation'][:200]}..."
            )

        # Get analysis summary
        analyses = engine.list_analyses()
        print(f"\n📊 Analysis Summary:")
        print(f"   Total analyses: {len(analyses)}")
        print(
            f"   Average quality: {sum(a['quality_score'] for a in analyses) / len(analyses):.2f}"
        )

        return {
            "exercise": "10.4",
            "success": True,
            "codes_analyzed": len(analysis_results),
            "generation_success": "error" not in generation_result,
            "explanation_success": "error" not in explanation_result,
            "average_quality": sum(r.quality_score for r in analysis_results)
            / len(analysis_results),
        }

    except Exception as e:
        logger.error(f"Exercise 10.4 failed: {e}")
        return {"exercise": "10.4", "success": False, "error": str(e)}


async def exercise_10_5_conversational_ai():
    """
    Exercise 10.5: Conversational AI Platform

    This exercise demonstrates:
    - Advanced conversation management
    - Context-aware dialogue systems
    - Multi-turn conversation optimization
    - Conversation analytics and insights
    """
    print("\n" + "=" * 60)
    print("Exercise 10.5: Conversational AI Platform")
    print("=" * 60)

    try:
        # Initialize conversational AI platform
        platform = ConversationalAIPlatform(":memory:")  # Use in-memory DB for testing

        # Start a conversation
        conv_result = await platform.start_conversation(
            user_id="test_user_001", mode="educational", title="AI Learning Session"
        )

        if not conv_result["success"]:
            raise Exception(f"Failed to start conversation: {conv_result['error']}")

        conversation_id = conv_result["conversation_id"]
        print(f"✅ Started conversation: {conversation_id}")

        # Simulate multi-turn conversation
        conversation_turns = [
            "Hello! I'm interested in learning about machine learning.",
            "What's the difference between supervised and unsupervised learning?",
            "Can you give me an example of a supervised learning algorithm?",
            "How do I know which algorithm to choose for my problem?",
            "Thank you for the explanations. This has been very helpful!",
        ]

        print(f"\n💬 Simulating {len(conversation_turns)}-turn conversation...")

        conversation_results = []
        for i, message in enumerate(conversation_turns, 1):
            print(f"\n👤 Turn {i}: {message}")

            response = await platform.send_message(conversation_id, message)
            conversation_results.append(response)

            if response["success"]:
                print(f"🤖 Response: {response['response'][:150]}...")
                print(f"   Confidence: {response['confidence']:.2f}")
                print(f"   Processing time: {response['processing_time']:.2f}s")
            else:
                print(f"❌ Error: {response['error']}")

        # End conversation and get analytics
        print(f"\n🏁 Ending conversation...")
        end_result = await platform.end_conversation(conversation_id)

        if end_result["success"]:
            analytics = end_result["final_analytics"]
            print(f"✅ Conversation ended successfully")
            print(f"   Total messages: {analytics['message_count']}")
            print(f"   Quality score: {analytics['quality_score']:.2f}")
            print(f"   Engagement score: {analytics['engagement_score']:.2f}")

        # Get conversation history
        history = platform.get_conversation_history(conversation_id)
        if history["success"]:
            print(f"📜 Conversation history: {len(history['messages'])} messages")
            print(
                f"   Topics discussed: {', '.join(history['context']['topics_discussed'])}"
            )

        # Get user analytics
        user_analytics = platform.get_user_analytics("test_user_001")
        if user_analytics["success"]:
            print(f"\n📊 User Analytics:")
            print(f"   Total conversations: {user_analytics['total_conversations']}")
            print(f"   Average quality: {user_analytics['average_quality_score']:.2f}")

        # Get system analytics
        system_analytics = platform.get_system_analytics()
        if system_analytics["success"]:
            print(f"\n🔍 System Analytics:")
            print(f"   Total conversations: {system_analytics['total_conversations']}")
            print(
                f"   Average response time: {system_analytics['average_response_time']:.2f}s"
            )

        successful_turns = len(
            [r for r in conversation_results if r.get("success", False)]
        )

        return {
            "exercise": "10.5",
            "success": True,
            "conversation_turns": len(conversation_turns),
            "successful_turns": successful_turns,
            "conversation_ended": end_result.get("success", False),
            "final_quality": (
                analytics.get("quality_score", 0) if end_result.get("success") else 0
            ),
        }

    except Exception as e:
        logger.error(f"Exercise 10.5 failed: {e}")
        return {"exercise": "10.5", "success": False, "error": str(e)}


async def run_all_exercises():
    """Run all Module 10 exercises and return comprehensive results"""
    print("🚀 DSPy Module 10: Advanced Projects - Complete Solutions")
    print("=" * 80)

    # Setup DSPy environment
    if not setup_dspy_environment():
        print("❌ Failed to setup DSPy environment")
        return {"success": False, "error": "Environment setup failed"}

    # Run all exercises
    exercises = [
        exercise_10_1_multi_agent_system,
        exercise_10_2_research_system,
        exercise_10_3_document_processing,
        exercise_10_4_code_analysis,
        exercise_10_5_conversational_ai,
    ]

    results = []
    successful_exercises = 0

    for exercise_func in exercises:
        try:
            print(f"\n🔄 Running {exercise_func.__name__}...")
            result = await exercise_func()
            results.append(result)
            if result.get("success", False):
                successful_exercises += 1
                print(f"✅ {exercise_func.__name__} completed successfully")
            else:
                print(f"❌ {exercise_func.__name__} failed")
        except Exception as e:
            logger.error(f"Exercise {exercise_func.__name__} failed: {e}")
            results.append(
                {"exercise": exercise_func.__name__, "success": False, "error": str(e)}
            )

    # Summary
    print("\n" + "=" * 80)
    print("MODULE 10 SUMMARY")
    print("=" * 80)
    print(f"Total Exercises: {len(exercises)}")
    print(f"Successful: {successful_exercises}")
    print(f"Failed: {len(exercises) - successful_exercises}")
    print(f"Success Rate: {(successful_exercises / len(exercises)) * 100:.1f}%")

    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
        exercise_name = result.get("exercise", "Unknown")
        print(f"  Exercise {exercise_name}: {status}")
        if not result.get("success", False) and "error" in result:
            print(f"    Error: {result['error']}")

    print("\n🎯 Advanced Capabilities Demonstrated:")
    print("  • Multi-agent system coordination and communication")
    print("  • Intelligent research automation and synthesis")
    print("  • Multi-format document processing and analysis")
    print("  • Code analysis, generation, and refactoring")
    print("  • Advanced conversational AI with memory and context")
    print("  • Real-time analytics and performance monitoring")
    print("  • Production-ready error handling and recovery")
    print("  • Scalable architecture patterns and best practices")

    return {
        "success": successful_exercises == len(exercises),
        "total_exercises": len(exercises),
        "successful_exercises": successful_exercises,
        "results": results,
    }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="DSPy Module 10 Advanced Projects Solutions"
    )
    parser.add_argument(
        "--exercise", type=str, help="Run specific exercise (10.1, 10.2, etc.)"
    )
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    if args.exercise:
        exercise_map = {
            "10.1": exercise_10_1_multi_agent_system,
            "10.2": exercise_10_2_research_system,
            "10.3": exercise_10_3_document_processing,
            "10.4": exercise_10_4_code_analysis,
            "10.5": exercise_10_5_conversational_ai,
        }

        if args.exercise in exercise_map:
            setup_dspy_environment()
            asyncio.run(exercise_map[args.exercise]())
        else:
            print(f"Unknown exercise: {args.exercise}")
            print(f"Available exercises: {', '.join(exercise_map.keys())}")

    elif args.all or len(sys.argv) == 1:
        asyncio.run(run_all_exercises())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
