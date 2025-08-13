# Module 08: Custom DSPy Modules & Components - Examples

This directory contains comprehensive Python script examples for Module 08, demonstrating advanced custom DSPy module development, component composition, testing, and workflow orchestration.

## Examples Overview

### Example 1: Basic Custom Module Creation

**File:** `basic_custom_module_creation.py`

**Learning Objectives:**

- Create custom DSPy modules with proper inheritance
- Implement required methods and validation
- Add comprehensive metadata and documentation
- Integrate DSPy functionality with fallback mechanisms

**Key Features:**

- `BasicTextAnalyzer` class with multiple analysis modes
- Comprehensive input validation and error handling
- Performance tracking and metrics collection
- Integration with DSPy ChainOfThought for enhanced analysis
- Fallback mechanisms for robust operation

**Usage:**

```bash
uv run basic_custom_module_creation.py
```

### Example 2: Component Composition and Pipeline Creation

**File:** `component_composition_patterns.py`

**Learning Objectives:**

- Create complex processing pipelines from simple components
- Implement different composition patterns
- Handle data flow between components
- Add error handling and recovery mechanisms

**Key Features:**

- Custom components: `KeywordExtractorComponent`, `TextStatisticsComponent`
- Advanced pipeline: `AdvancedTextProcessingPipeline`
- Conditional routing: `ConditionalTextRouter`
- Parallel execution demonstrations
- Performance optimization techniques

**Usage:**

```bash
uv run component_composition_patterns.py
```

### Example 3: Module Testing and Validation

**File:** `module_testing_strategies.py`

**Learning Objectives:**

- Create comprehensive test suites for custom modules
- Implement automated testing and validation workflows
- Perform performance benchmarking and load testing
- Assess module quality and reliability

**Key Features:**

- `TestableTextProcessor` for testing demonstrations
- Comprehensive test suite creation with edge cases
- Unit testing, integration testing, and performance benchmarking
- Quality assessment and automated test generation
- Detailed test reporting and analysis

**Usage:**

```bash
uv run module_testing_strategies.py
```

### Example 4: Workflow Orchestration and Management

**File:** `workflow_orchestration_demo.py`

**Learning Objectives:**

- Create and manage complex workflows with multiple execution modes
- Implement workflow configuration and parameter management
- Handle workflow execution monitoring and error recovery
- Serialize and share workflows and modules

**Key Features:**

- Custom modules: `DocumentAnalysisModule`, `ReportGeneratorModule`
- Complex workflow configurations with sequential, parallel, and conditional execution
- Workflow monitoring and status tracking
- Configuration management and serialization
- Module export/import capabilities
- Performance optimization through parallel execution

**Usage:**

```bash
uv run workflow_orchestration_demo.py
```

## Prerequisites

Before running the examples, ensure you have:

1. **Python Environment:** Python 3.8 or higher
2. **Dependencies:** Install required packages using uv:

   ```bash
   uv add dspy pyyaml
   ```

3. **DSPy Language Model Configuration (Optional):**
   The examples use fallback implementations when DSPy language models are not configured. This is intentional and allows the examples to run without requiring API keys or local model setup. You'll see informational messages about fallback usage - this is normal behavior.
   
   To use actual DSPy language model functionality, configure a model before running:
   ```python
   import dspy
   
   # Latest models (recommended)
   dspy.configure(lm=dspy.LM(model="openai/gpt-5", api_key="your-api-key"))
   # or
   dspy.configure(lm=dspy.LM(model="anthropic/claude-3-7-sonnet-20250219", api_key="your-api-key"))
   
   # Local models
   dspy.configure(lm=dspy.LM(model="ollama/llama3.1"))
   ```

4. **Module Structure:** Ensure the parent module files are available:
   - `custom_module_template.py`
   - `component_library.py`
   - `module_testing_framework.py`
   - `module_composition.py`

## Running the Examples

Each example is a standalone Python script that can be executed independently using `uv run`:

```bash
# Navigate to the examples directory
cd 08-custom-modules/examples

# Run individual examples
uv run basic_custom_module_creation.py
uv run component_composition_patterns.py
uv run module_testing_strategies.py
uv run workflow_orchestration_demo.py
```

## Expected Output

Each example provides comprehensive demonstrations with:

- **Detailed explanations** of concepts and implementations
- **Step-by-step execution** with progress indicators
- **Performance metrics** and timing information
- **Error handling demonstrations** with various edge cases
- **Best practices** and optimization techniques
- **Key learning points** summarized at the end

## Generated Files

Some examples create additional files during execution:

### Workflow Orchestration Demo Generated Files

- `sample_workflow.json` - Workflow configuration example
- `exported_module.json` - Serialized module example

These files demonstrate configuration management and module serialization capabilities.

## Key Learning Outcomes

After running these examples, you will understand:

### Module Development

- How to create robust custom DSPy modules
- Proper inheritance and method implementation
- Metadata management and documentation
- Performance tracking and optimization

### Component Composition

- Building complex systems from simple components
- Different composition patterns and their use cases
- Data flow management between components
- Error handling and recovery strategies

### Testing and Validation

- Comprehensive testing strategies for DSPy modules
- Automated test generation and execution
- Performance benchmarking and load testing
- Quality assessment and reliability evaluation

### Workflow Orchestration

- Complex workflow creation and management
- Multiple execution modes (sequential, parallel, conditional)
- Configuration management and serialization
- Module sharing and deployment strategies

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all parent module files are in the correct location
   - Check Python path configuration

2. **DSPy Integration Issues:**
   - Verify DSPy installation: `uv add dspy-ai`
   - Check for proper DSPy configuration with current models (gpt-4o, gpt-4-turbo, claude-3-5-sonnet, etc.)

3. **Performance Issues:**
   - Some examples include intentional delays for demonstration
   - Adjust timing parameters if needed for your environment

4. **File Permission Issues:**
   - Ensure write permissions for configuration file generation
   - Check directory permissions for temporary files

### Getting Help

If you encounter issues:

1. Check the error messages and logs
2. Verify all prerequisites are met
3. Review the code comments for additional context
4. Ensure proper module imports and dependencies

## Advanced Usage

### Customization Options

Each example includes parameters that can be modified:

- **Analysis modes** in Basic Custom Module Creation
- **Component configurations** in Component Composition Patterns
- **Test parameters** in Module Testing Strategies
- **Workflow configurations** in Workflow Orchestration Demo

### Extension Opportunities

These examples provide a foundation for:

- Creating domain-specific modules
- Building production-ready pipelines
- Implementing custom testing frameworks
- Developing workflow orchestration systems

## Best Practices Demonstrated

1. **Code Organization:** Clear separation of concerns and modular design
2. **Error Handling:** Comprehensive error handling and recovery mechanisms
3. **Documentation:** Extensive documentation and code comments
4. **Testing:** Multiple testing strategies and validation approaches
5. **Performance:** Optimization techniques and performance monitoring
6. **Maintainability:** Clean code practices and extensible architectures

## Next Steps

After running these examples, consider:

1. **Extending the modules** with additional functionality
2. **Creating domain-specific components** for your use cases
3. **Building production workflows** using the orchestration patterns
4. **Contributing improvements** to the framework
5. **Exploring advanced DSPy features** in subsequent modules

These examples provide a comprehensive foundation for advanced DSPy module development and serve as reference implementations for best practices in the field.
