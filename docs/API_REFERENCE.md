# DSPy Learning Framework - API Reference

## Overview

This comprehensive API reference covers all modules, classes, and functions in the DSPy Learning Framework. The framework provides a complete learning path from basic DSPy concepts to advanced production-ready AI systems.

## Table of Contents

- [Core Configuration](#core-configuration)
- [Module 1: Foundations](#module-1-foundations)
- [Module 2: Signatures and Fields](#module-2-signatures-and-fields)
- [Module 3: DSPy Modules](#module-3-dspy-modules)
- [Module 4: Optimization Techniques](#module-4-optimization-techniques)
- [Module 5: Evaluation and Metrics](#module-5-evaluation-and-metrics)
- [Module 6: Advanced Patterns](#module-6-advanced-patterns)
- [Module 7: Integration Patterns](#module-7-integration-patterns)
- [Module 8: Custom Modules](#module-8-custom-modules)
- [Module 9: Production Deployment](#module-9-production-deployment)
- [Module 10: Advanced Projects](#module-10-advanced-projects)
- [Utilities and Helpers](#utilities-and-helpers)

---

## Core Configuration

### DSPy Configuration (`08-custom-modules/dspy_config.py`)

The core configuration module handles automatic DSPy setup with live models.

#### Functions

**Method: `configure_dspy_lm(provider: str = "auto") -> bool`**

Configures DSPy with the best available language model.

**Parameters:**

- `provider` (str): Model provider - "auto", "openai", "anthropic", or "fallback"

**Returns:**

- `bool`: True if successfully configured, False if using fallback

**Example:**

```python
from dspy_config import configure_dspy_lm, get_configured_model_info

# Auto-configure with best available model
success = configure_dspy_lm("auto")
if success:
    model_info = get_configured_model_info()
    print(f"Using model: {model_info['model']}")
```

**Supported Models:**

- **OpenAI**: `gpt-4.1` (primary)
- **Anthropic**: `claude-3-7-sonnet-20250219`

**Method: `is_dspy_configured() -> bool`**

Checks if DSPy is properly configured and ready to use.

##### `get_configured_model_info() -> dict`

Returns information about the currently configured model.

**Returns:**

```python
{
    "configured": bool,
    "model": str,
    "provider": str,
    "adapter": str  # if applicable
}
```

---

## Module 1: Foundations

### Basic Signatures

**Class: `BasicSignature`**

Simple question-answering signature demonstrating core DSPy concepts.

```python
class BasicSignature(dspy.Signature):
    question: str = dspy.InputField(desc="Question to be answered")
    answer: str = dspy.OutputField(desc="Answer to the question")

# Usage
predictor = dspy.Predict(BasicSignature)
result = predictor(question="What is the capital of France?")
print(result.answer)
```

**Class: `DetailedQASignature`**

Enhanced question-answering with context and confidence scoring.

```python
class DetailedQASignature(dspy.Signature):
    question: str = dspy.InputField(desc="Question to be answered")
    context: str = dspy.InputField(desc="Context information for answering")
    answer: str = dspy.OutputField(desc="Detailed answer based on context")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1)")

# Usage with ChainOfThought
qa_module = dspy.ChainOfThought(DetailedQASignature)
result = qa_module(
    question="What programming language is DSPy built with?",
    context="DSPy is a framework implemented in Python..."
)
```

---

## Module 2: Signatures and Fields

### Advanced Field Types

#### Input Fields

- `dspy.InputField(desc="description")` - Standard input field
- `dspy.InputField(desc="description", format="json")` - Structured input
- `dspy.InputField(desc="description", prefix="Context:")` - Prefixed input

#### Output Fields

- `dspy.OutputField(desc="description")` - Standard output field
- `dspy.OutputField(desc="description", format="list")` - List output
- `dspy.OutputField(desc="description", format="json")` - JSON output

### Complex Signatures

**Class: `MultiModalSignature`**

Handles multiple input types and structured outputs.

```python
class MultiModalSignature(dspy.Signature):
    text_input: str = dspy.InputField(desc="Primary text input")
    context_data: str = dspy.InputField(desc="Additional context", format="json")
    parameters: str = dspy.InputField(desc="Processing parameters")
    
    primary_output: str = dspy.OutputField(desc="Main result")
    metadata: str = dspy.OutputField(desc="Result metadata", format="json")
    confidence: float = dspy.OutputField(desc="Confidence score")
```

---

## Module 3: DSPy Modules

### Core Modules

**Method: `dspy.Predict`**

Basic prediction module for simple input-output tasks.

```python
# Simple prediction
predictor = dspy.Predict(YourSignature)
result = predictor(input_field="your input")
```

#### `dspy.ChainOfThought`

Reasoning module that shows intermediate thinking steps.

```python
# Chain of thought reasoning
reasoner = dspy.ChainOfThought(YourSignature)
result = reasoner(input_field="complex question")
# Access reasoning: result.rationale
```

#### `dspy.ReAct`

Reasoning and acting module for complex problem-solving.

```python
# ReAct pattern
reactor = dspy.ReAct(YourSignature)
result = reactor(input_field="problem to solve")
```

### Custom Module Patterns

#### Conditional Modules

```python
class ConditionalModule(dspy.Module):
    def __init__(self):
        self.simple_predictor = dspy.Predict(SimpleSignature)
        self.complex_reasoner = dspy.ChainOfThought(ComplexSignature)
    
    def forward(self, input_text, complexity="simple"):
        if complexity == "simple":
            return self.simple_predictor(input_text=input_text)
        else:
            return self.complex_reasoner(input_text=input_text)
```

---

## Module 4: Optimization Techniques

### Optimizers

**Class: `BootstrapFewShot`**

Automatic few-shot example generation and optimization.

```python
from dspy.teleprompt import BootstrapFewShot

# Setup optimizer
optimizer = BootstrapFewShot(
    metric=your_metric_function,
    max_bootstrapped_demos=8,
    max_labeled_demos=16
)

# Optimize module
optimized_module = optimizer.compile(
    student=your_module,
    trainset=training_data
)
```

#### `MIPRO` (Multi-Prompt Optimization)

Advanced multi-prompt optimization for complex tasks.

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=your_metric,
    num_candidates=10,
    init_temperature=1.0
)

optimized_module = optimizer.compile(
    student=your_module,
    trainset=training_data,
    valset=validation_data
)
```

### Custom Metrics

#### Creating Evaluation Metrics

```python
def accuracy_metric(example, pred, trace=None):
    """Custom accuracy metric"""
    return example.answer.lower() == pred.answer.lower()

def semantic_similarity_metric(example, pred, trace=None):
    """Semantic similarity metric"""
    # Implementation using embeddings or other similarity measures
    similarity_score = calculate_similarity(example.answer, pred.answer)
    return similarity_score > 0.8
```

---

## Module 5: Evaluation and Metrics

### Evaluation Framework

#### `Evaluate`

Comprehensive evaluation system for DSPy modules.

```python
from dspy import Evaluate

# Setup evaluation
evaluator = Evaluate(
    devset=test_dataset,
    metric=your_metric_function,
    num_threads=4,
    display_progress=True
)

# Run evaluation
results = evaluator(your_module)
print(f"Accuracy: {results}")
```

### Built-in Metrics

#### Classification Metrics

```python
# Accuracy
def accuracy(example, pred, trace=None):
    return example.label == pred.label

# F1 Score
def f1_score(example, pred, trace=None):
    # Implementation for F1 calculation
    return calculate_f1(example.label, pred.label)
```

#### Generation Metrics

```python
# BLEU Score
def bleu_score(example, pred, trace=None):
    return calculate_bleu(example.reference, pred.generated_text)

# Rouge Score
def rouge_score(example, pred, trace=None):
    return calculate_rouge(example.reference, pred.generated_text)
```

---

## Module 6: Advanced Patterns

### Multi-Agent Systems

**Class: `BaseAgent`**

Abstract base class for creating intelligent agents.

```python
from multi_agent_system import BaseAgent, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus):
        super().__init__(agent_id, AgentRole.RESEARCHER, message_bus)
        self.custom_module = dspy.ChainOfThought(YourSignature)
    
    async def process_task(self, task):
        result = self.custom_module(task_description=task.description)
        return {"status": "completed", "result": result}
```

**Class: `MultiAgentResearchSystem`**

Coordinated multi-agent research system.

```python
from multi_agent_system import MultiAgentResearchSystem

# Initialize system
system = MultiAgentResearchSystem()
await system.initialize_system()
await system.start_system()

# Conduct research
research_id = await system.conduct_research(
    topic="Your research topic",
    objectives=["objective1", "objective2"]
)

# Get results
results = await system.get_research_results(research_id)
```

### Workflow Patterns

#### `ResearchWorkflow`

Automated research workflow management.

```python
from research_workflow import ResearchWorkflow

workflow = ResearchWorkflow("research_workflow")
workflow.add_step("information_gathering", gather_info_module)
workflow.add_step("analysis", analysis_module)
workflow.add_step("synthesis", synthesis_module)

result = await workflow.execute(input_data)
```

---

## Module 7: Integration Patterns

### External API Integration

**Class: `APIIntegrationModule`**

Base class for integrating external APIs with DSPy.

```python
class CustomAPIModule(dspy.Module):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.processor = dspy.ChainOfThought(ProcessingSignature)
    
    def forward(self, query: str):
        # Call external API
        api_result = self.call_external_api(query)
        
        # Process with DSPy
        processed = self.processor(
            raw_data=api_result,
            query=query
        )
        
        return processed
```

### Database Integration

**Class: `DatabaseModule`**

Integration with databases for persistent storage.

```python
class DatabaseModule(dspy.Module):
    def __init__(self, db_connection):
        self.db = db_connection
        self.query_processor = dspy.ChainOfThought(QuerySignature)
    
    def forward(self, natural_query: str):
        # Convert natural language to SQL
        sql_result = self.query_processor(
            natural_query=natural_query,
            schema_info=self.get_schema()
        )
        
        # Execute query
        results = self.db.execute(sql_result.sql_query)
        return results
```

---

## Module 8: Custom Modules

### Advanced Custom Modules

**Class: `ConditionalReasoningModule`**

Module with conditional logic and multiple reasoning paths.

```python
class ConditionalReasoningModule(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassificationSignature)
        self.simple_reasoner = dspy.Predict(SimpleReasoningSignature)
        self.complex_reasoner = dspy.ChainOfThought(ComplexReasoningSignature)
    
    def forward(self, problem: str):
        # Classify problem complexity
        classification = self.classifier(problem=problem)
        
        # Route to appropriate reasoner
        if classification.complexity == "simple":
            return self.simple_reasoner(problem=problem)
        else:
            return self.complex_reasoner(problem=problem)
```

**Class: `IterativeRefinementModule`**

Module that iteratively refines its output.

```python
class IterativeRefinementModule(dspy.Module):
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations
        self.generator = dspy.ChainOfThought(GenerationSignature)
        self.critic = dspy.Predict(CriticSignature)
        self.refiner = dspy.ChainOfThought(RefinementSignature)
    
    def forward(self, input_text: str):
        current_output = self.generator(input_text=input_text)
        
        for i in range(self.max_iterations):
            critique = self.critic(
                input_text=input_text,
                current_output=current_output.output
            )
            
            if critique.needs_refinement == "no":
                break
            
            current_output = self.refiner(
                input_text=input_text,
                current_output=current_output.output,
                critique=critique.feedback
            )
        
        return current_output
```

---

## Module 9: Production Deployment

### Deployment Utilities

**Class: `ProductionOptimizer`**

Optimizes DSPy modules for production deployment.

```python
from production_deployment import ProductionOptimizer

optimizer = ProductionOptimizer()

# Optimize for production
production_module = optimizer.optimize_for_production(
    module=your_module,
    target_latency=2.0,  # seconds
    target_accuracy=0.95,
    optimization_budget=100  # API calls
)
```

**Class: `ScalingManager`**

Manages scaling and load balancing for DSPy applications.

```python
from scaling_strategies import ScalingManager

scaling_manager = ScalingManager(
    max_concurrent_requests=100,
    auto_scaling_enabled=True,
    scaling_metrics=['latency', 'throughput']
)

# Deploy with scaling
await scaling_manager.deploy_module(your_module)
```

### Monitoring and Maintenance

**Class: `PerformanceMonitor`**

Real-time performance monitoring for production systems.

```python
from maintenance_operations import PerformanceMonitor

monitor = PerformanceMonitor()

# Monitor module performance
@monitor.track_performance
def your_production_function(input_data):
    return your_module(input_data)

# Get performance metrics
metrics = monitor.get_metrics()
```

#### `OperationalTools`

Tools for managing production DSPy systems.

```python
from operational_tools import OperationalTools

ops = OperationalTools()

# Health check
health_status = ops.health_check(your_module)

# Performance analysis
performance_report = ops.analyze_performance(
    module=your_module,
    time_window="24h"
)

# Auto-optimization
optimized_module = ops.auto_optimize(
    module=your_module,
    performance_target=0.95
)
```

---

## Module 10: Advanced Projects

### Multi-Agent Research System

**Class: `MultiAgentResearchSystem`**

Complete multi-agent research platform.

```python
from multi_agent_system import MultiAgentResearchSystem

# Initialize system
system = MultiAgentResearchSystem()
await system.initialize_system()
await system.start_system()

# Conduct research
research_id = await system.conduct_research(
    topic="Artificial Intelligence in Healthcare",
    objectives=[
        "Identify current applications",
        "Analyze market trends",
        "Evaluate future opportunities"
    ]
)

# Monitor progress
status = await system.get_research_results(research_id)
```

### Integrated Research Platform

**Class: `IntegratedResearchPlatform`**

Comprehensive research platform with workflow management.

```python
from integrated_research_system import IntegratedResearchPlatform

# Initialize platform
platform = IntegratedResearchPlatform()
await platform.initialize_platform()
await platform.start_platform()

# Create research project
project_id = await platform.create_research_project(
    project_name="Market Analysis",
    description="Comprehensive market analysis project",
    objectives=["Market size", "Competition analysis", "Growth projections"],
    template_name="market_research"
)

# Start project
success = await platform.start_research_project(project_id)

# Get results
results = await platform.get_project_status(project_id)
```

### Document Processing System

**`DocumentProcessingEngine`**

Advanced document processing with AI-powered analysis.

```python
from document_processing_system import DocumentProcessingEngine

# Initialize engine
engine = DocumentProcessingEngine()

# Process single document
result = await engine.process_document("path/to/document.pdf")

print(f"Category: {result.category.value}")
print(f"Quality Score: {result.quality_score}")
print(f"Summary: {result.content.summary}")
print(f"Key Phrases: {result.content.key_phrases}")

# Batch processing
results = await engine.process_batch([
    "doc1.pdf", "doc2.docx", "doc3.txt"
])

# Search processed documents
search_results = engine.search_documents(
    query="artificial intelligence",
    filters={"category": "research_paper", "min_quality": 0.8}
)
```

### Code Analysis Tool

**Class: `CodeAnalysisEngine`**

Comprehensive code analysis and generation system.

```python
from code_analysis_tool import CodeAnalysisEngine, CodeLanguage

# Initialize engine
engine = CodeAnalysisEngine()

# Analyze code
result = await engine.analyze_code(
    code=your_code_string,
    file_path="example.py"
)

print(f"Language: {result.language.value}")
print(f"Quality Score: {result.quality_score}")
print(f"Issues: {len(result.issues)}")
print(f"Suggestions: {len(result.suggestions)}")

# Generate code
generation_result = await engine.generate_code(
    requirements="Create a Python function for sorting algorithms",
    language=CodeLanguage.PYTHON
)

# Explain code
explanation = await engine.explain_code(
    code=your_code,
    explanation_level="detailed"
)

# Refactor code
refactoring = await engine.refactor_code(
    code=your_code,
    refactoring_goals="improve performance and readability"
)
```

### Conversational AI Platform

**Class: `ConversationalAIPlatform`**

Advanced conversational AI with memory and context management.

```python
from conversational_ai_platform import ConversationalAIPlatform

# Initialize platform
platform = ConversationalAIPlatform()

# Start conversation
conv_result = await platform.start_conversation(
    user_id="user123",
    mode="educational",
    title="Learning Session"
)

conversation_id = conv_result["conversation_id"]

# Send messages
response = await platform.send_message(
    conversation_id=conversation_id,
    message="Hello! Can you help me learn about AI?"
)

print(f"Response: {response['response']}")
print(f"Confidence: {response['confidence']}")

# Get conversation history
history = platform.get_conversation_history(conversation_id)

# End conversation
end_result = await platform.end_conversation(conversation_id)
analytics = end_result["final_analytics"]
```

---

## Utilities and Helpers

### DSPy Helpers (`solutions/utilities/dspy_helpers.py`)

**Method: `setup_dspy_environment(provider="auto")`**

Sets up DSPy environment with live models.

```python
from solutions.utilities.dspy_helpers import setup_dspy_environment

# Setup with auto-detection
success = setup_dspy_environment("auto")

# Setup with specific provider
success = setup_dspy_environment("openai")
```

#### `create_test_signature(input_fields, output_fields, signature_name)`

Dynamically creates DSPy signatures for testing.

```python
from solutions.utilities.dspy_helpers import create_test_signature

# Create custom signature
TestSignature = create_test_signature(
    input_fields={"question": "Question to answer"},
    output_fields={"answer": "Answer to question"},
    signature_name="CustomQA"
)

# Use the signature
predictor = dspy.Predict(TestSignature)
result = predictor(question="What is DSPy?")
```

#### `benchmark_signature_performance(signature_class, test_cases)`

Benchmarks signature performance across different module types.

```python
from solutions.utilities.dspy_helpers import benchmark_signature_performance

# Benchmark performance
results = benchmark_signature_performance(
    signature_class=YourSignature,
    test_cases=[
        {"input_field": "test case 1"},
        {"input_field": "test case 2"}
    ],
    module_types=["ChainOfThought", "Predict"],
    iterations=3
)

print(f"Best module: {results['summary']['best_performing_module']}")
print(f"Success rate: {results['summary']['best_success_rate']}")
```

### Evaluation Helpers

**Method: `comprehensive_evaluation(module, dataset, metrics)`**

Runs comprehensive evaluation with multiple metrics.

```python
from solutions.utilities.evaluation_helpers import comprehensive_evaluation

# Run comprehensive evaluation
results = comprehensive_evaluation(
    module=your_module,
    dataset=test_dataset,
    metrics=[accuracy_metric, f1_metric, bleu_metric]
)

print(f"Overall score: {results['overall_score']}")
print(f"Metric breakdown: {results['metric_scores']}")
```

---

## Best Practices

### 1. Signature Design

- Use descriptive field names and descriptions
- Include format specifications for structured outputs
- Consider confidence scores for uncertain outputs

### 2. Module Composition

- Start with simple modules and compose complex ones
- Use conditional logic for different processing paths
- Implement proper error handling and fallbacks

### 3. Optimization

- Always evaluate before and after optimization
- Use appropriate metrics for your specific task
- Consider multiple optimization strategies

### 4. Production Deployment

- Monitor performance continuously
- Implement proper logging and error tracking
- Use scaling strategies for high-load scenarios

### 5. Testing and Validation

- Create comprehensive test suites
- Validate with live models before deployment
- Use proper evaluation metrics for your domain

---

## Common Patterns

### 1. Question-Answering Pipeline

```python
class QAPipeline(dspy.Module):
    def __init__(self):
        self.retriever = dspy.Retrieve(k=5)
        self.qa_module = dspy.ChainOfThought(QASignature)
    
    def forward(self, question):
        context = self.retriever(question)
        answer = self.qa_module(question=question, context=context)
        return answer
```

### 2. Multi-Step Reasoning

```python
class MultiStepReasoner(dspy.Module):
    def __init__(self):
        self.step1 = dspy.ChainOfThought(Step1Signature)
        self.step2 = dspy.ChainOfThought(Step2Signature)
        self.synthesizer = dspy.ChainOfThought(SynthesisSignature)
    
    def forward(self, problem):
        result1 = self.step1(problem=problem)
        result2 = self.step2(problem=problem, context=result1.output)
        final = self.synthesizer(step1=result1.output, step2=result2.output)
        return final
```

### 3. Conditional Processing

```python
class ConditionalProcessor(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassificationSignature)
        self.simple_processor = dspy.Predict(SimpleSignature)
        self.complex_processor = dspy.ChainOfThought(ComplexSignature)
    
    def forward(self, input_text):
        classification = self.classifier(text=input_text)
        
        if classification.complexity == "simple":
            return self.simple_processor(text=input_text)
        else:
            return self.complex_processor(text=input_text)
```

---

## Error Handling

### Common Error Patterns

```python
try:
    result = your_module(input_data)
except dspy.DSPyError as e:
    # Handle DSPy-specific errors
    logger.error(f"DSPy error: {e}")
    return fallback_response
except Exception as e:
    # Handle general errors
    logger.error(f"Unexpected error: {e}")
    return error_response
```

### Validation Patterns

```python
def validate_output(result, expected_format):
    """Validate module output format"""
    if not hasattr(result, 'answer'):
        raise ValueError("Missing required 'answer' field")
    
    if expected_format == "json":
        try:
            json.loads(result.answer)
        except json.JSONDecodeError:
            raise ValueError("Answer is not valid JSON")
    
    return True
```

---

This API reference provides comprehensive documentation for all components of the DSPy Learning Framework. For specific implementation details, refer to the individual module files and example code in the repository.
