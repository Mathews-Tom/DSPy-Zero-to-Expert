#!/usr/bin/env python3
"""
Comprehensive Research Assistant System

This is a complete, production-ready implementation of an intelligent research
assistant that demonstrates advanced DSPy patterns and real-world application
architecture. The system integrates multiple AI agents, document processing,
and knowledge synthesis capabilities.

Features:
- Multi-agent research coordination
- Intelligent document analysis and synthesis
- Real-time research workflow automation
- Advanced query processing and optimization
- Comprehensive evaluation and monitoring

Architecture:
- Modular design with clear separation of concerns
- Async/await patterns for scalable performance
- Comprehensive error handling and recovery
- Production-ready logging and monitoring
- Extensible plugin architecture

Author: DSPy Learning Framework
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from document_analyzer import DocumentAnalyzer
from evaluation_engine import EvaluationEngine
from knowledge_synthesizer import KnowledgeSynthesizer
from query_processor import QueryProcessor
from research_coordinator import ResearchCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("research_assistant.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ResearchAssistantSystem:
    """
    Main Research Assistant System

    This class orchestrates all components of the research assistant,
    providing a unified interface for research operations and maintaining
    system state and configuration.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the research assistant system"""
        self.config = config or self._default_config()

        # Initialize core components
        self.query_processor = QueryProcessor(self.config.get("query_processor", {}))
        self.research_coordinator = ResearchCoordinator(
            self.config.get("coordinator", {})
        )
        self.document_analyzer = DocumentAnalyzer(
            self.config.get("document_analyzer", {})
        )
        self.knowledge_synthesizer = KnowledgeSynthesizer(
            self.config.get("synthesizer", {})
        )
        self.evaluation_engine = EvaluationEngine(self.config.get("evaluation", {}))

        # System state
        self.active_research_sessions = {}
        self.system_metrics = {
            "total_queries": 0,
            "successful_research_sessions": 0,
            "average_response_time": 0.0,
            "knowledge_base_size": 0,
        }

        logger.info("Research Assistant System initialized")

    def _default_config(self) -> Dict:
        """Return default system configuration"""
        return {
            "max_concurrent_sessions": 10,
            "default_research_depth": "comprehensive",
            "enable_real_time_learning": True,
            "quality_threshold": 0.8,
            "max_response_time": 300,  # 5 minutes
            "query_processor": {
                "max_query_length": 1000,
                "enable_query_expansion": True,
                "similarity_threshold": 0.7,
            },
            "coordinator": {
                "max_agents": 5,
                "coordination_strategy": "collaborative",
                "task_timeout": 180,
            },
            "document_analyzer": {
                "supported_formats": ["pdf", "txt", "html", "md"],
                "max_document_size": 10 * 1024 * 1024,  # 10MB
                "enable_ocr": True,
            },
            "synthesizer": {
                "synthesis_strategy": "hierarchical",
                "max_sources": 50,
                "confidence_threshold": 0.6,
            },
            "evaluation": {
                "enable_continuous_evaluation": True,
                "evaluation_metrics": ["relevance", "completeness", "accuracy"],
                "feedback_integration": True,
            },
        }

    async def process_research_query(
        self,
        query: str,
        user_id: str = "default",
        research_options: Optional[Dict] = None,
    ) -> Dict:
        """
        Process a research query end-to-end

        Args:
            query: Research query string
            user_id: User identifier for session management
            research_options: Optional research configuration

        Returns:
            Dict containing research results and metadata
        """
        session_id = f"{user_id}_{len(self.active_research_sessions)}"
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(
                f"Starting research session {session_id} for query: {query[:100]}..."
            )

            # Step 1: Process and expand query
            processed_query = await self.query_processor.process_query(
                query=query, user_context={"user_id": user_id}, options=research_options
            )

            if not processed_query["success"]:
                raise Exception(f"Query processing failed: {processed_query['error']}")

            # Step 2: Coordinate research activities
            research_plan = await self.research_coordinator.create_research_plan(
                processed_query=processed_query,
                research_depth=research_options.get("depth", "comprehensive"),
            )

            # Step 3: Execute research plan
            research_results = await self.research_coordinator.execute_research_plan(
                research_plan=research_plan, session_id=session_id
            )

            # Step 4: Analyze collected documents
            if research_results.get("documents"):
                analysis_results = await self.document_analyzer.analyze_documents(
                    documents=research_results["documents"],
                    analysis_focus=processed_query["expanded_query"],
                )
                research_results["document_analysis"] = analysis_results

            # Step 5: Synthesize knowledge
            synthesis_result = await self.knowledge_synthesizer.synthesize_knowledge(
                research_data=research_results,
                synthesis_goals=processed_query["research_goals"],
            )

            # Step 6: Evaluate results
            evaluation_result = await self.evaluation_engine.evaluate_research_output(
                query=query,
                research_output=synthesis_result,
                quality_metrics=self.config["evaluation"]["evaluation_metrics"],
            )

            # Compile final results
            execution_time = asyncio.get_event_loop().time() - start_time

            final_result = {
                "session_id": session_id,
                "success": True,
                "query": query,
                "processed_query": processed_query,
                "research_results": research_results,
                "synthesis": synthesis_result,
                "evaluation": evaluation_result,
                "execution_time": execution_time,
                "metadata": {
                    "sources_consulted": len(research_results.get("sources", [])),
                    "documents_analyzed": len(research_results.get("documents", [])),
                    "confidence_score": evaluation_result.get("confidence_score", 0.0),
                    "quality_score": evaluation_result.get("quality_score", 0.0),
                },
            }

            # Update system metrics
            self._update_system_metrics(final_result)

            # Store session
            self.active_research_sessions[session_id] = final_result

            logger.info(
                f"Research session {session_id} completed successfully in {execution_time:.2f}s"
            )
            return final_result

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_result = {
                "session_id": session_id,
                "success": False,
                "query": query,
                "error": str(e),
                "execution_time": execution_time,
            }

            logger.error(f"Research session {session_id} failed: {e}")
            return error_result

    def _update_system_metrics(self, result: Dict):
        """Update system performance metrics"""
        self.system_metrics["total_queries"] += 1

        if result.get("success"):
            self.system_metrics["successful_research_sessions"] += 1

        # Update average response time
        current_avg = self.system_metrics["average_response_time"]
        total_queries = self.system_metrics["total_queries"]
        new_time = result.get("execution_time", 0)

        self.system_metrics["average_response_time"] = (
            current_avg * (total_queries - 1) + new_time
        ) / total_queries

    async def get_research_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a research session by ID"""
        return self.active_research_sessions.get(session_id)

    async def list_research_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """List research sessions, optionally filtered by user"""
        sessions = []
        for session_id, session_data in self.active_research_sessions.items():
            if user_id is None or session_id.startswith(f"{user_id}_"):
                sessions.append(
                    {
                        "session_id": session_id,
                        "query": session_data.get("query", ""),
                        "success": session_data.get("success", False),
                        "execution_time": session_data.get("execution_time", 0),
                        "quality_score": session_data.get("metadata", {}).get(
                            "quality_score", 0
                        ),
                    }
                )
        return sessions

    def get_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
        success_rate = 0.0
        if self.system_metrics["total_queries"] > 0:
            success_rate = (
                self.system_metrics["successful_research_sessions"]
                / self.system_metrics["total_queries"]
            )

        return {
            **self.system_metrics,
            "success_rate": success_rate,
            "active_sessions": len(self.active_research_sessions),
        }

    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Research Assistant System...")

        # Save session data if needed
        # Close database connections
        # Clean up resources

        logger.info("Research Assistant System shutdown complete")


async def demo_research_assistant():
    """Demonstrate the research assistant system capabilities"""
    print("🔬 Research Assistant System Demo")
    print("=" * 60)

    # Initialize system
    system = ResearchAssistantSystem()

    # Demo queries
    demo_queries = [
        "What are the latest developments in quantum computing for machine learning?",
        "How is artificial intelligence being used in healthcare diagnostics?",
        "What are the environmental impacts of large language models?",
        "What are the current challenges in autonomous vehicle technology?",
    ]

    print(f"🚀 Processing {len(demo_queries)} research queries...")

    results = []
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📋 Query {i}: {query}")

        # Process query
        result = await system.process_research_query(
            query=query,
            user_id=f"demo_user_{i}",
            research_options={
                "depth": "comprehensive",
                "max_sources": 10,
                "include_recent_only": True,
            },
        )

        results.append(result)

        if result["success"]:
            print(f"✅ Research completed successfully")
            print(f"   Execution time: {result['execution_time']:.2f}s")
            print(f"   Sources consulted: {result['metadata']['sources_consulted']}")
            print(f"   Quality score: {result['metadata']['quality_score']:.2f}")
            print(
                f"   Synthesis preview: {result['synthesis'].get('summary', 'N/A')[:150]}..."
            )
        else:
            print(f"❌ Research failed: {result['error']}")

    # Show system metrics
    print(f"\n📊 System Performance Metrics:")
    metrics = system.get_system_metrics()
    print(f"   Total queries processed: {metrics['total_queries']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Average response time: {metrics['average_response_time']:.2f}s")
    print(f"   Active sessions: {metrics['active_sessions']}")

    # List sessions
    sessions = await system.list_research_sessions()
    print(f"\n📋 Research Sessions:")
    for session in sessions:
        status = "✅" if session["success"] else "❌"
        print(f"   {status} {session['session_id']}: {session['query'][:50]}...")

    # Shutdown
    await system.shutdown()

    print(f"\n🎯 Demo completed! Processed {len(results)} queries")
    successful = len([r for r in results if r["success"]])
    print(
        f"   Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)"
    )


async def interactive_research_session():
    """Run an interactive research session"""
    print("🔬 Interactive Research Assistant")
    print("=" * 60)
    print("Enter research queries (type 'quit' to exit)")

    system = ResearchAssistantSystem()
    session_count = 0

    try:
        while True:
            query = input("\n🔍 Research Query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            session_count += 1
            print(f"\n🔄 Processing query {session_count}...")

            result = await system.process_research_query(
                query=query,
                user_id="interactive_user",
                research_options={"depth": "standard"},
            )

            if result["success"]:
                print(f"\n✅ Research Results:")
                print(f"📊 Quality Score: {result['metadata']['quality_score']:.2f}")
                print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
                print(f"📚 Sources: {result['metadata']['sources_consulted']}")

                if "synthesis" in result and "summary" in result["synthesis"]:
                    print(f"\n📝 Summary:")
                    print(result["synthesis"]["summary"])

                if "synthesis" in result and "key_findings" in result["synthesis"]:
                    print(f"\n🔍 Key Findings:")
                    for i, finding in enumerate(
                        result["synthesis"]["key_findings"][:3], 1
                    ):
                        print(f"   {i}. {finding}")
            else:
                print(f"\n❌ Research failed: {result['error']}")

    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user")

    finally:
        await system.shutdown()
        print(f"\n📊 Session Summary: {session_count} queries processed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Research Assistant System")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive mode"
    )
    parser.add_argument("--query", type=str, help="Process single query")

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo_research_assistant())
    elif args.interactive:
        asyncio.run(interactive_research_session())
    elif args.query:

        async def single_query():
            system = ResearchAssistantSystem()
            result = await system.process_research_query(args.query, "cli_user")

            if result["success"]:
                print("✅ Research completed successfully")
                print(f"Quality Score: {result['metadata']['quality_score']:.2f}")
                print(f"Summary: {result['synthesis'].get('summary', 'N/A')}")
            else:
                print(f"❌ Research failed: {result['error']}")

            await system.shutdown()

        asyncio.run(single_query())
    else:
        print("🔬 Research Assistant System")
        print("Use --demo for demonstration or --interactive for interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
