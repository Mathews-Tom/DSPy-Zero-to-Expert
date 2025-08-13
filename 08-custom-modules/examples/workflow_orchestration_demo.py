#!/usr/bin/env python3
"""
Workflow Orchestration and Management Demo

This example demonstrates advanced workflow orchestration, including complex workflow
creation, execution management, configuration handling, and module serialization.

Learning Objectives:
- Create and manage complex workflows with multiple execution modes
- Implement workflow configuration and parameter management
- Handle workflow execution monitoring and error recovery
- Serialize and share workflows and modules
- Optimize workflow performance and resource usage

Author: DSPy Learning Framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
import time
from typing import Any, Dict, List

from component_library import (
    ComponentPipeline,
    SentimentAnalyzerComponent,
    TextClassifierComponent,
    TextCleanerComponent,
    TextSummarizerComponent,
)
from custom_module_template import CustomModuleBase, ModuleMetadata
from module_composition import (
    ConfigurationManager,
    ExecutionContext,
    ModuleConfiguration,
    WorkflowConfiguration,
    WorkflowExecutionMode,
    WorkflowOrchestrator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentAnalysisModule(CustomModuleBase):
    """Specialized module for document analysis workflows"""

    def __init__(self, analysis_depth: str = "standard"):
        metadata = ModuleMetadata(
            name="Document Analysis Module",
            description="Comprehensive document analysis for workflow orchestration",
            version="1.0.0",
            author="DSPy Learning Framework",
            tags=["document", "analysis", "workflow"],
        )
        super().__init__(metadata)

        self.analysis_depth = analysis_depth
        self._initialized = True

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Analyze document content"""
        text = kwargs.get("text", "")

        if not text:
            return {"error": "No text provided for analysis"}

        # Basic document metrics
        words = text.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        analysis = {
            "document_length": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "analysis_depth": self.analysis_depth,
        }

        # Depth-specific analysis
        if self.analysis_depth == "detailed":
            analysis.update(self._detailed_analysis(text, words))
        elif self.analysis_depth == "summary":
            analysis.update(self._summary_analysis(text, words, sentences))

        # Document classification
        analysis["document_type"] = self._classify_document(text)
        analysis["complexity_score"] = self._calculate_complexity(words, sentences)

        return analysis

    def _detailed_analysis(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Perform detailed document analysis"""
        import string

        # Character analysis
        char_analysis = {
            "letters": sum(1 for c in text if c.isalpha()),
            "digits": sum(1 for c in text if c.isdigit()),
            "punctuation": sum(1 for c in text if c in string.punctuation),
            "whitespace": sum(1 for c in text if c.isspace()),
        }

        # Vocabulary analysis
        unique_words = set(word.lower().strip(string.punctuation) for word in words)
        vocab_analysis = {
            "unique_words": len(unique_words),
            "vocabulary_richness": len(unique_words) / len(words) if words else 0,
            "average_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
        }

        return {
            "character_analysis": char_analysis,
            "vocabulary_analysis": vocab_analysis,
        }

    def _summary_analysis(
        self, text: str, words: List[str], sentences: List[str]
    ) -> Dict[str, Any]:
        """Perform summary document analysis"""
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        return {
            "average_sentence_length": avg_sentence_length,
            "readability_estimate": (
                "easy"
                if avg_sentence_length < 15
                else "medium" if avg_sentence_length < 25 else "difficult"
            ),
            "has_questions": "?" in text,
            "has_exclamations": "!" in text,
        }

    def _classify_document(self, text: str) -> str:
        """Simple document type classification"""
        text_lower = text.lower()

        # Simple keyword-based classification
        if any(
            word in text_lower
            for word in ["research", "study", "analysis", "methodology"]
        ):
            return "academic"
        elif any(
            word in text_lower for word in ["company", "business", "market", "revenue"]
        ):
            return "business"
        elif any(
            word in text_lower
            for word in ["technology", "software", "algorithm", "system"]
        ):
            return "technical"
        elif any(
            word in text_lower for word in ["story", "character", "plot", "narrative"]
        ):
            return "creative"
        else:
            return "general"

    def _calculate_complexity(self, words: List[str], sentences: List[str]) -> float:
        """Calculate document complexity score (0-1)"""
        if not words or not sentences:
            return 0.0

        # Factors contributing to complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)

        # Normalize and combine factors
        word_complexity = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
        sentence_complexity = min(avg_sentence_length / 30, 1.0)  # Normalize to 0-1

        return (word_complexity + sentence_complexity) / 2


class ReportGeneratorModule(CustomModuleBase):
    """Module for generating comprehensive analysis reports"""

    def __init__(self, report_format: str = "detailed"):
        metadata = ModuleMetadata(
            name="Report Generator Module",
            description="Generate comprehensive analysis reports from workflow results",
            version="1.0.0",
            author="DSPy Learning Framework",
            tags=["report", "generator", "workflow"],
        )
        super().__init__(metadata)

        self.report_format = report_format
        self._initialized = True

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Generate analysis report from workflow data"""
        # Collect all available data from workflow context
        report_data = {}

        # Extract data from various sources
        for key, value in kwargs.items():
            if isinstance(value, dict):
                report_data.update(value)
            else:
                report_data[key] = value

        # Generate report based on format
        if self.report_format == "summary":
            report = self._generate_summary_report(report_data)
        elif self.report_format == "detailed":
            report = self._generate_detailed_report(report_data)
        else:
            report = self._generate_basic_report(report_data)

        return {
            "report": report,
            "report_format": self.report_format,
            "generated_at": time.time(),
            "data_sources": list(report_data.keys()),
        }

    def _generate_summary_report(self, data: Dict[str, Any]) -> str:
        """Generate a summary report"""
        lines = ["=== ANALYSIS SUMMARY REPORT ===", ""]

        # Document metrics
        if "word_count" in data:
            lines.append(f"Document Length: {data['word_count']} words")
        if "sentence_count" in data:
            lines.append(f"Sentences: {data['sentence_count']}")

        # Sentiment analysis
        if "sentiment" in data:
            lines.append(f"Sentiment: {data['sentiment'].title()}")
            if "confidence" in data:
                lines.append(f"Confidence: {data['confidence']:.2f}")

        # Document type
        if "document_type" in data:
            lines.append(f"Document Type: {data['document_type'].title()}")

        # Summary
        if "summary" in data:
            lines.extend(["", "Summary:", data["summary"]])

        return "\n".join(lines)

    def _generate_detailed_report(self, data: Dict[str, Any]) -> str:
        """Generate a detailed report"""
        lines = ["=== DETAILED ANALYSIS REPORT ===", ""]

        # Document Overview
        lines.append("DOCUMENT OVERVIEW:")
        if "word_count" in data:
            lines.append(f"  Word Count: {data['word_count']}")
        if "sentence_count" in data:
            lines.append(f"  Sentence Count: {data['sentence_count']}")
        if "paragraph_count" in data:
            lines.append(f"  Paragraph Count: {data['paragraph_count']}")
        if "document_type" in data:
            lines.append(f"  Document Type: {data['document_type'].title()}")
        if "complexity_score" in data:
            lines.append(f"  Complexity Score: {data['complexity_score']:.2f}")

        lines.append("")

        # Sentiment Analysis
        if "sentiment" in data:
            lines.append("SENTIMENT ANALYSIS:")
            lines.append(f"  Sentiment: {data['sentiment'].title()}")
            if "confidence" in data:
                lines.append(f"  Confidence: {data['confidence']:.2f}")
            if "reasoning" in data:
                lines.append(f"  Reasoning: {data['reasoning']}")

        lines.append("")

        # Content Analysis
        if "keywords" in data:
            lines.append("CONTENT ANALYSIS:")
            lines.append(f"  Keywords: {', '.join(data['keywords'][:10])}")

        if "vocabulary_analysis" in data:
            vocab = data["vocabulary_analysis"]
            lines.append("VOCABULARY ANALYSIS:")
            lines.append(f"  Unique Words: {vocab.get('unique_words', 'N/A')}")
            lines.append(
                f"  Vocabulary Richness: {vocab.get('vocabulary_richness', 0):.2f}"
            )

        # Summary
        if "summary" in data:
            lines.extend(["", "SUMMARY:", data["summary"]])

        return "\n".join(lines)

    def _generate_basic_report(self, data: Dict[str, Any]) -> str:
        """Generate a basic report"""
        lines = ["=== BASIC ANALYSIS REPORT ===", ""]

        # Key metrics only
        if "word_count" in data:
            lines.append(f"Words: {data['word_count']}")
        if "sentiment" in data:
            lines.append(f"Sentiment: {data['sentiment']}")
        if "document_type" in data:
            lines.append(f"Type: {data['document_type']}")

        return "\n".join(lines)


def create_document_analysis_workflow() -> WorkflowConfiguration:
    """Create a comprehensive document analysis workflow"""

    workflow_config = WorkflowConfiguration(
        workflow_id="document_analysis_workflow",
        name="Comprehensive Document Analysis",
        description="Complete document analysis including cleaning, sentiment, classification, and reporting",
        version="1.0.0",
        execution_mode=WorkflowExecutionMode.SEQUENTIAL,
        modules=[
            ModuleConfiguration(
                module_id="clean_document",
                module_type="text_cleaner",
                input_mappings={"text": "input_document"},
                output_mappings={"cleaned_text": "cleaned_document"},
                parameters={"remove_html": True, "normalize_whitespace": True},
            ),
            ModuleConfiguration(
                module_id="analyze_document",
                module_type="document_analyzer",
                input_mappings={"text": "cleaned_document"},
                output_mappings={
                    "word_count": "doc_word_count",
                    "sentence_count": "doc_sentence_count",
                    "document_type": "doc_type",
                    "complexity_score": "doc_complexity",
                },
                parameters={"analysis_depth": "detailed"},
            ),
            ModuleConfiguration(
                module_id="analyze_sentiment",
                module_type="sentiment_analyzer",
                input_mappings={"text": "cleaned_document"},
                output_mappings={
                    "sentiment": "doc_sentiment",
                    "confidence": "sentiment_confidence",
                    "reasoning": "sentiment_reasoning",
                },
            ),
            ModuleConfiguration(
                module_id="classify_document",
                module_type="document_classifier",
                input_mappings={"text": "cleaned_document"},
                output_mappings={"predicted_category": "doc_category"},
                parameters={
                    "categories": [
                        "business",
                        "technical",
                        "academic",
                        "creative",
                        "general",
                    ]
                },
            ),
            ModuleConfiguration(
                module_id="summarize_document",
                module_type="text_summarizer",
                input_mappings={"text": "cleaned_document"},
                output_mappings={"summary": "doc_summary"},
                parameters={"max_length": 200},
            ),
            ModuleConfiguration(
                module_id="generate_report",
                module_type="report_generator",
                input_mappings={
                    "word_count": "doc_word_count",
                    "sentence_count": "doc_sentence_count",
                    "document_type": "doc_type",
                    "sentiment": "doc_sentiment",
                    "confidence": "sentiment_confidence",
                    "summary": "doc_summary",
                },
                output_mappings={"report": "final_report"},
                parameters={"report_format": "detailed"},
            ),
        ],
        global_parameters={
            "processing_mode": "comprehensive",
            "include_metadata": True,
        },
        error_handling={
            "stop_on_error": False,
            "retry_failed_steps": True,
            "max_retries": 2,
        },
    )

    return workflow_config


def create_parallel_analysis_workflow() -> WorkflowConfiguration:
    """Create a parallel analysis workflow for performance comparison"""

    workflow_config = WorkflowConfiguration(
        workflow_id="parallel_analysis_workflow",
        name="Parallel Document Analysis",
        description="Parallel execution of independent analysis components",
        version="1.0.0",
        execution_mode=WorkflowExecutionMode.PARALLEL,
        modules=[
            ModuleConfiguration(
                module_id="sentiment_analysis",
                module_type="sentiment_analyzer",
                input_mappings={"text": "input_document"},
                output_mappings={"sentiment": "parallel_sentiment"},
            ),
            ModuleConfiguration(
                module_id="document_classification",
                module_type="document_classifier",
                input_mappings={"text": "input_document"},
                output_mappings={"predicted_category": "parallel_category"},
            ),
            ModuleConfiguration(
                module_id="document_analysis",
                module_type="document_analyzer",
                input_mappings={"text": "input_document"},
                output_mappings={"complexity_score": "parallel_complexity"},
            ),
        ],
        global_parameters={"parallel_execution": True, "max_workers": 3},
    )

    return workflow_config


def create_conditional_workflow() -> WorkflowConfiguration:
    """Create a conditional workflow that adapts based on document characteristics"""

    workflow_config = WorkflowConfiguration(
        workflow_id="conditional_analysis_workflow",
        name="Conditional Document Analysis",
        description="Adaptive analysis based on document characteristics",
        version="1.0.0",
        execution_mode=WorkflowExecutionMode.CONDITIONAL,
        modules=[
            ModuleConfiguration(
                module_id="initial_analysis",
                module_type="document_analyzer",
                input_mappings={"text": "input_document"},
                output_mappings={
                    "word_count": "doc_length",
                    "document_type": "doc_type",
                },
            ),
            ModuleConfiguration(
                module_id="detailed_analysis",
                module_type="document_analyzer",
                input_mappings={"text": "input_document"},
                output_mappings={"complexity_score": "detailed_complexity"},
                parameters={"analysis_depth": "detailed"},
                conditions={
                    "variable_exists": "doc_length",
                    "variable_greater_than": ("doc_length", 100),
                },
            ),
            ModuleConfiguration(
                module_id="technical_processing",
                module_type="document_classifier",
                input_mappings={"text": "input_document"},
                output_mappings={"predicted_category": "tech_category"},
                conditions={"variable_equals": ("doc_type", "technical")},
            ),
            ModuleConfiguration(
                module_id="summary_generation",
                module_type="text_summarizer",
                input_mappings={"text": "input_document"},
                output_mappings={"summary": "conditional_summary"},
                conditions={"variable_greater_than": ("doc_length", 200)},
            ),
        ],
    )

    return workflow_config


def demonstrate_workflow_creation_and_execution():
    """Demonstrate creating and executing complex workflows"""
    print("=== Workflow Orchestration and Management Demo ===\n")
    print("1. Workflow Creation and Execution:")
    print("-" * 50)

    # Create orchestrator and register modules
    orchestrator = WorkflowOrchestrator()

    # Register all required modules
    orchestrator.register_module("text_cleaner", TextCleanerComponent())
    orchestrator.register_module("sentiment_analyzer", SentimentAnalyzerComponent())
    orchestrator.register_module("text_summarizer", TextSummarizerComponent())
    orchestrator.register_module(
        "document_classifier",
        TextClassifierComponent(
            ["business", "technical", "academic", "creative", "general"]
        ),
    )
    orchestrator.register_module(
        "document_analyzer", DocumentAnalysisModule(analysis_depth="detailed")
    )
    orchestrator.register_module(
        "report_generator", ReportGeneratorModule(report_format="detailed")
    )

    # Create and register workflow
    workflow_config = create_document_analysis_workflow()
    orchestrator.register_workflow(workflow_config)

    # Test document
    test_document = """
    Artificial Intelligence and Machine Learning Technologies
    
    The field of artificial intelligence has experienced unprecedented growth in recent years.
    Machine learning algorithms, particularly deep neural networks, have achieved remarkable
    success across various domains including natural language processing, computer vision,
    and robotics.
    
    Modern AI systems demonstrate sophisticated capabilities in pattern recognition,
    decision making, and autonomous operation. However, challenges remain in areas such as
    explainability, bias mitigation, and ethical AI development.
    
    The integration of AI technologies into business processes has transformed industries
    ranging from healthcare and finance to transportation and entertainment. Organizations
    are increasingly leveraging AI to optimize operations, enhance customer experiences,
    and drive innovation.
    """

    # Execute workflow
    print("Executing comprehensive document analysis workflow...")

    workflow_inputs = {"input_document": test_document}
    result = orchestrator.execute_workflow(
        "document_analysis_workflow", workflow_inputs
    )

    print(f"Workflow Execution Results:")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Steps Completed: {result['performance_metrics']['steps_completed']}")
    print(f"  Success Rate: {result['performance_metrics']['success_rate']:.2%}")
    print(f"  Total Time: {result['performance_metrics']['total_execution_time']:.3f}s")

    # Display final report if available
    if result["status"] == "completed" and "result" in result:
        workflow_result = result["result"]
        if "final_variables" in workflow_result:
            final_vars = workflow_result["final_variables"]
            if "final_report" in final_vars:
                print(f"\nGenerated Report:")
                print("-" * 30)
                print(final_vars["final_report"])


def demonstrate_parallel_execution():
    """Demonstrate parallel workflow execution"""
    print("\n2. Parallel Workflow Execution:")
    print("-" * 50)

    orchestrator = WorkflowOrchestrator()

    # Register modules
    orchestrator.register_module("sentiment_analyzer", SentimentAnalyzerComponent())
    orchestrator.register_module(
        "document_classifier",
        TextClassifierComponent(
            ["business", "technical", "academic", "creative", "general"]
        ),
    )
    orchestrator.register_module("document_analyzer", DocumentAnalysisModule())

    # Create and register parallel workflow
    parallel_config = create_parallel_analysis_workflow()
    orchestrator.register_workflow(parallel_config)

    # Test with same document
    test_document = "This is a technical document about software development and system architecture."

    print("Executing parallel analysis workflow...")

    start_time = time.time()
    result = orchestrator.execute_workflow(
        "parallel_analysis_workflow", {"input_document": test_document}
    )
    parallel_time = time.time() - start_time

    print(f"Parallel Execution Results:")
    print(f"  Status: {result['status']}")
    print(f"  Execution Time: {parallel_time:.3f}s")
    print(f"  Steps Completed: {result['performance_metrics']['steps_completed']}")

    if result["status"] == "completed":
        workflow_result = result["result"]
        if "steps" in workflow_result:
            print(f"  Parallel Steps:")
            for step in workflow_result["steps"]:
                if step.get("success"):
                    print(f"    {step['module_type']}: {step['execution_time']:.3f}s")


def demonstrate_conditional_execution():
    """Demonstrate conditional workflow execution"""
    print("\n3. Conditional Workflow Execution:")
    print("-" * 50)

    orchestrator = WorkflowOrchestrator()

    # Register modules
    orchestrator.register_module("document_analyzer", DocumentAnalysisModule())
    orchestrator.register_module(
        "document_classifier",
        TextClassifierComponent(
            ["business", "technical", "academic", "creative", "general"]
        ),
    )
    orchestrator.register_module("text_summarizer", TextSummarizerComponent())

    # Create and register conditional workflow
    conditional_config = create_conditional_workflow()
    orchestrator.register_workflow(conditional_config)

    # Test with different document types and lengths
    test_documents = [
        ("Short Document", "Brief text."),
        (
            "Long Technical Document",
            "This is a comprehensive technical document about software architecture and system design. "
            * 20,
        ),
        (
            "Business Document",
            "Our company's quarterly revenue has increased significantly due to improved market strategies and customer engagement initiatives.",
        ),
    ]

    for doc_name, doc_text in test_documents:
        print(f"\nTesting: {doc_name}")
        print(f"Length: {len(doc_text.split())} words")

        result = orchestrator.execute_workflow(
            "conditional_analysis_workflow", {"input_document": doc_text}
        )

        print(f"  Status: {result['status']}")
        print(f"  Steps Completed: {result['performance_metrics']['steps_completed']}")

        if result["status"] == "completed":
            workflow_result = result["result"]
            if "steps" in workflow_result:
                executed_steps = [
                    step["module_id"]
                    for step in workflow_result["steps"]
                    if step.get("executed", True)
                ]
                print(f"  Executed Steps: {', '.join(executed_steps)}")


def demonstrate_configuration_management():
    """Demonstrate workflow configuration management"""
    print("\n4. Configuration Management:")
    print("-" * 50)

    config_manager = ConfigurationManager()

    # Create a sample workflow configuration
    sample_config = create_document_analysis_workflow()

    # Save configuration to file
    config_file = "08-custom-modules/solutions/sample_workflow.json"

    try:
        config_manager.save_configuration(sample_config, config_file)
        print(f"✅ Configuration saved to {config_file}")

        # Load configuration back
        loaded_config = config_manager.load_configuration(config_file)
        print(f"✅ Configuration loaded successfully")
        print(f"   Workflow: {loaded_config.name}")
        print(f"   Modules: {len(loaded_config.modules)}")
        print(f"   Execution Mode: {loaded_config.execution_mode.value}")

        # Verify configuration integrity
        print(f"   Configuration Integrity:")
        print(f"     Original ID: {sample_config.workflow_id}")
        print(f"     Loaded ID: {loaded_config.workflow_id}")
        print(
            f"     Match: {'✅' if sample_config.workflow_id == loaded_config.workflow_id else '❌'}"
        )

    except Exception as e:
        print(f"❌ Configuration management failed: {e}")


def demonstrate_workflow_monitoring():
    """Demonstrate workflow execution monitoring"""
    print("\n5. Workflow Execution Monitoring:")
    print("-" * 50)

    orchestrator = WorkflowOrchestrator()

    # Register modules
    orchestrator.register_module("text_cleaner", TextCleanerComponent())
    orchestrator.register_module("sentiment_analyzer", SentimentAnalyzerComponent())
    orchestrator.register_module("document_analyzer", DocumentAnalysisModule())

    # Create a simple workflow for monitoring
    monitoring_config = WorkflowConfiguration(
        workflow_id="monitoring_test_workflow",
        name="Monitoring Test Workflow",
        description="Simple workflow for monitoring demonstration",
        execution_mode=WorkflowExecutionMode.SEQUENTIAL,
        modules=[
            ModuleConfiguration(
                module_id="clean_step",
                module_type="text_cleaner",
                input_mappings={"text": "input_text"},
                output_mappings={"cleaned_text": "cleaned"},
            ),
            ModuleConfiguration(
                module_id="analyze_step",
                module_type="document_analyzer",
                input_mappings={"text": "cleaned"},
                output_mappings={"word_count": "words"},
            ),
            ModuleConfiguration(
                module_id="sentiment_step",
                module_type="sentiment_analyzer",
                input_mappings={"text": "cleaned"},
                output_mappings={"sentiment": "final_sentiment"},
            ),
        ],
    )

    orchestrator.register_workflow(monitoring_config)

    # Execute workflow and monitor
    print("Starting workflow execution with monitoring...")

    test_input = {
        "input_text": "This is a test document for monitoring workflow execution."
    }
    result = orchestrator.execute_workflow("monitoring_test_workflow", test_input)

    execution_id = result["execution_id"]

    # Get execution status
    status = orchestrator.get_execution_status(execution_id)

    if status:
        print(f"Execution Monitoring:")
        print(f"  Execution ID: {status['execution_id']}")
        print(f"  Workflow ID: {status['workflow_id']}")
        print(f"  Status: {status['status']}")
        print(f"  Current Step: {status['current_step']}")
        print(f"  Elapsed Time: {status['elapsed_time']:.3f}s")
        print(f"  Step Results: {status['step_results_count']}")
        print(f"  Errors: {status['error_count']}")

    # Display execution context details
    if execution_id in orchestrator.execution_contexts:
        context = orchestrator.execution_contexts[execution_id]
        print(f"\nExecution Context Details:")
        print(f"  Variables: {len(context.variables)} items")
        print(f"  Step Results: {len(context.step_results)} steps")
        print(f"  Performance Metrics: {context.performance_metrics}")


def demonstrate_module_serialization():
    """Demonstrate module export and import capabilities"""
    print("\n6. Module Serialization and Sharing:")
    print("-" * 50)

    config_manager = ConfigurationManager()

    # Create a custom module for export
    custom_analyzer = DocumentAnalysisModule(analysis_depth="detailed")

    # Export module
    export_file = "08-custom-modules/solutions/exported_module.json"

    try:
        config_manager.export_module(custom_analyzer, export_file)
        print(f"✅ Module exported to {export_file}")

        # Import module back
        imported_module = config_manager.import_module(export_file)
        print(f"✅ Module imported successfully")

        # Test imported module
        test_result = imported_module(text="Test document for imported module.")
        print(f"   Imported module test:")
        print(f"     Word count: {test_result.get('word_count', 'N/A')}")
        print(f"     Document type: {test_result.get('document_type', 'N/A')}")
        print(f"     Analysis depth: {test_result.get('analysis_depth', 'N/A')}")

    except Exception as e:
        print(f"❌ Module serialization failed: {e}")


def demonstrate_performance_optimization():
    """Demonstrate workflow performance optimization"""
    print("\n7. Performance Optimization:")
    print("-" * 50)

    # Compare sequential vs parallel execution
    orchestrator = WorkflowOrchestrator()

    # Register modules
    orchestrator.register_module("sentiment_analyzer", SentimentAnalyzerComponent())
    orchestrator.register_module(
        "document_classifier",
        TextClassifierComponent(["business", "technical", "academic"]),
    )
    orchestrator.register_module("document_analyzer", DocumentAnalysisModule())

    # Create sequential workflow
    sequential_config = WorkflowConfiguration(
        workflow_id="sequential_perf_test",
        name="Sequential Performance Test",
        description="Sequential execution for performance comparison",
        execution_mode=WorkflowExecutionMode.SEQUENTIAL,
        modules=[
            ModuleConfiguration(
                "step1", "sentiment_analyzer", input_mappings={"text": "input"}
            ),
            ModuleConfiguration(
                "step2", "document_classifier", input_mappings={"text": "input"}
            ),
            ModuleConfiguration(
                "step3", "document_analyzer", input_mappings={"text": "input"}
            ),
        ],
    )

    # Create parallel workflow
    parallel_config = WorkflowConfiguration(
        workflow_id="parallel_perf_test",
        name="Parallel Performance Test",
        description="Parallel execution for performance comparison",
        execution_mode=WorkflowExecutionMode.PARALLEL,
        modules=[
            ModuleConfiguration(
                "step1", "sentiment_analyzer", input_mappings={"text": "input"}
            ),
            ModuleConfiguration(
                "step2", "document_classifier", input_mappings={"text": "input"}
            ),
            ModuleConfiguration(
                "step3", "document_analyzer", input_mappings={"text": "input"}
            ),
        ],
    )

    orchestrator.register_workflow(sequential_config)
    orchestrator.register_workflow(parallel_config)

    # Performance test
    test_document = (
        "This is a comprehensive business document about market analysis and strategic planning. "
        * 10
    )
    test_input = {"input": test_document}

    # Test sequential execution
    sequential_times = []
    for i in range(3):
        start_time = time.time()
        result = orchestrator.execute_workflow("sequential_perf_test", test_input)
        sequential_times.append(time.time() - start_time)

    # Test parallel execution
    parallel_times = []
    for i in range(3):
        start_time = time.time()
        result = orchestrator.execute_workflow("parallel_perf_test", test_input)
        parallel_times.append(time.time() - start_time)

    # Compare results
    avg_sequential = sum(sequential_times) / len(sequential_times)
    avg_parallel = sum(parallel_times) / len(parallel_times)
    speedup = avg_sequential / avg_parallel if avg_parallel > 0 else 0

    print(f"Performance Comparison:")
    print(f"  Sequential Average: {avg_sequential:.3f}s")
    print(f"  Parallel Average: {avg_parallel:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(
        f"  Performance Gain: {((avg_sequential - avg_parallel) / avg_sequential * 100):.1f}%"
    )


if __name__ == "__main__":
    """
    Workflow Orchestration and Management Demo

    This script demonstrates:
    1. Complex workflow creation with multiple execution modes
    2. Workflow execution monitoring and status tracking
    3. Configuration management and serialization
    4. Module export/import capabilities
    5. Performance optimization through parallel execution
    6. Conditional workflow execution based on data characteristics
    7. Comprehensive workflow reporting and analysis
    """

    try:
        demonstrate_workflow_creation_and_execution()
        demonstrate_parallel_execution()
        demonstrate_conditional_execution()
        demonstrate_configuration_management()
        demonstrate_workflow_monitoring()
        demonstrate_module_serialization()
        demonstrate_performance_optimization()

        print("\n✅ Workflow Orchestration and Management demo completed successfully!")
        print("\nKey Learning Points:")
        print("- Workflow orchestration enables complex multi-step processing")
        print("- Different execution modes serve different performance and logic needs")
        print("- Configuration management enables workflow reusability and sharing")
        print("- Monitoring capabilities provide visibility into workflow execution")
        print("- Module serialization supports component sharing and deployment")
        print(
            "- Performance optimization through parallel execution can provide significant speedups"
        )
        print(
            "- Conditional workflows enable adaptive processing based on data characteristics"
        )
        print("- Comprehensive error handling ensures robust workflow execution")

    except Exception as e:
        print(f"\n❌ Workflow Orchestration and Management demo failed: {e}")
        logger.exception("Workflow Orchestration and Management demo execution failed")
