# DSPy x Marimo: Progressive Learning Repository Structure

## Course Overview

**Title:** DSPy x Marimo: Interactive Learning Repository  
**Description:** A progressive learning course that teaches DSPy concepts through Marimo-based interactive lessons  
**Target Audience:** Expert Python developers with GenAI and Agentic system experience  

### Prerequisites

- Advanced Python programming experience
- Knowledge of generative AI concepts and applications  
- Experience with agentic systems and LLM applications
- Familiarity with machine learning workflows
- Understanding of prompt engineering principles

### Learning Objectives

By the end of this course, you will be able to:

- Master all core DSPy components and modules
- Learn practical Marimo usage for interactive AI development
- Build advanced RAG and multi-step reasoning systems
- Understand DSPy optimization and evaluation techniques
- Create production-ready agentic applications

---

## Environment Setup Instructions

### Initial Setup with uv Package Manager

The course uses `uv` as the preferred package manager for its speed and modern Python project management capabilities.

#### 1. Install uv

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip/pipx
pip install uv
# or
pipx install uv
```

#### 2. Create Project Structure

```bash
# Clone or create the repository
gh repo clone Mathews-Tom/DSPy-Zero-to-Expert
cd DSPy-Zero-to-Expert

# Initialize uv project (if starting from scratch)
uv init --python 3.11
```

#### 3. Install Dependencies

Create a `pyproject.toml` with the following configuration:

```toml
[project]
name = "DSPy-Zero-to-Expert"
version = "0.1.0"
description = "Interactive DSPy learning through Marimo notebooks"
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "dspy-ai>=2.5.0",
    "marimo>=0.9.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "requests>=2.31.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "plotly>=5.15.0",
    "scikit-learn>=1.3.0",
    "datasets>=2.14.0",
    "transformers>=4.30.0",
    "faiss-cpu>=1.7.4",
    "chromadb>=0.4.0",
    "qdrant-client>=1.6.0",
    "mlflow>=2.8.0",
    "langfuse>=2.0.0",
    "tavily-python>=0.3.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "jupyter>=1.0.0",
    "jupytext>=1.15.0"
]

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "ipython>=8.14.0",
    "ipykernel>=6.25.0"
]
```

#### 4. Install and Setup Environment

```bash
# Install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Verify installation
uv run python -c "import dspy, marimo; print('Setup successful!')"
```

#### 5. Environment Configuration

Create a `.env` file for API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Other optional services
COHERE_API_KEY=your_cohere_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## Complete Module Outline

### Module 00: Environment Setup & Introduction

**Duration:** 45-60 minutes  
**Description:** Set up development environment with uv, understand the course structure, and create your first DSPy + Marimo application

**Learning Goals:**

- Install and configure uv package manager
- Set up DSPy and Marimo development environment
- Understand the synergy between DSPy and Marimo
- Create and run your first interactive DSPy notebook

**Key Concepts:** uv package management, Virtual environment setup, DSPy installation, Marimo basics, Interactive development

**Practical Exercises:**

- Install uv and create project structure
- Configure DSPy with multiple LLM providers
- Build a simple 'Hello DSPy' Marimo notebook
- Test reactive execution with parameter changes

---

### Module 01: DSPy Foundations: Signatures & Basic Modules

**Duration:** 60-75 minutes  
**Description:** Learn DSPy's core programming model through Signatures and basic modules, with hands-on Marimo interactions

**Learning Goals:**

- Master DSPy Signature creation and usage
- Understand the difference between inline and class-based signatures
- Work with dspy.Predict and dspy.ChainOfThought modules
- Build interactive signature testing with Marimo UI elements

**Key Concepts:** DSPy Signatures, Module architecture, Predict modules, ChainOfThought reasoning, Interactive parameter tuning

**Practical Exercises:**

- Create custom signatures for various NLP tasks
- Build interactive signature testers with Marimo sliders
- Compare Predict vs ChainOfThought performance
- Implement signature composition patterns

---

### Module 02: Advanced DSPy Modules: ReAct, Tools & Multi-Step Reasoning

**Duration:** 75-90 minutes  
**Description:** Explore advanced DSPy modules for complex reasoning, tool usage, and multi-step problem solving

**Learning Goals:**

- Implement ReAct (Reasoning + Acting) modules
- Integrate external tools and APIs with DSPy
- Build multi-step reasoning pipelines
- Create interactive debugging interfaces with Marimo

**Key Concepts:** ReAct methodology, Tool integration, Multi-step reasoning, Program of Thought, Interactive debugging

**Practical Exercises:**

- Build a ReAct agent with web search capabilities
- Create tool-augmented reasoning systems
- Implement multi-hop question answering
- Build debugging dashboards with Marimo

---

### Module 03: Retrieval-Augmented Generation with DSPy

**Duration:** 90-105 minutes  
**Description:** Master RAG implementation using DSPy modules, with vector databases and retrieval optimization

**Learning Goals:**

- Implement RAG pipelines with DSPy retrievers
- Work with vector databases and embedding models
- Optimize retrieval and generation components
- Build interactive RAG evaluation interfaces

**Key Concepts:** RAG architecture, Vector retrieval, Embedding optimization, Retrieval evaluation, Interactive RAG tuning

**Practical Exercises:**

- Build a complete RAG system with DSPy
- Implement custom retrievers and rankers
- Create RAG evaluation metrics
- Build interactive RAG parameter tuning interface

---

### Module 04: DSPy Optimization: Teleprompters & Automatic Tuning

**Duration:** 75-90 minutes  
**Description:** Learn DSPy's optimization framework to automatically improve prompts and model performance

**Learning Goals:**

- Understand DSPy optimization principles
- Use BootstrapFewShot and MIPRO optimizers
- Create custom metrics for optimization
- Visualize optimization progress with Marimo

**Key Concepts:** Teleprompter algorithms, Few-shot optimization, Metric design, Optimization visualization, Performance tracking

**Practical Exercises:**

- Optimize signatures with BootstrapFewShot
- Implement custom optimization metrics
- Build optimization progress dashboards
- Compare optimization strategies

---

### Module 05: Evaluation & Metrics: Measuring DSPy System Performance

**Duration:** 60-75 minutes  
**Description:** Design comprehensive evaluation frameworks and custom metrics for DSPy applications

**Learning Goals:**

- Design evaluation strategies for LLM systems
- Implement custom DSPy metrics
- Use evaluation for system improvement
- Build interactive evaluation dashboards

**Key Concepts:** Evaluation methodologies, Custom metrics, System benchmarking, Performance visualization, Iterative improvement

**Practical Exercises:**

- Create domain-specific evaluation metrics
- Build comprehensive evaluation pipelines
- Design interactive evaluation interfaces
- Implement A/B testing for DSPy systems

---

### Module 06: Working with Datasets & Examples in DSPy

**Duration:** 45-60 minutes  
**Description:** Learn to work with datasets, create examples, and manage training data for DSPy optimization

**Learning Goals:**

- Work with DSPy Example objects and datasets
- Create training and evaluation datasets
- Implement data preprocessing pipelines
- Build dataset exploration interfaces

**Key Concepts:** DSPy Examples, Dataset management, Data preprocessing, Interactive data exploration, Training data curation

**Practical Exercises:**

- Create custom dataset loaders
- Build data preprocessing pipelines
- Design dataset exploration interfaces
- Implement data quality metrics

---

### Module 07: Tracing, Debugging & Observability

**Duration:** 60-75 minutes  
**Description:** Master debugging techniques and observability tools for complex DSPy applications

**Learning Goals:**

- Use DSPy tracing and debugging tools
- Implement comprehensive logging strategies
- Build observability dashboards
- Debug complex multi-step pipelines

**Key Concepts:** DSPy tracing, Debug utilities, Observability patterns, Interactive debugging, Performance monitoring

**Practical Exercises:**

- Implement comprehensive tracing systems
- Build debugging dashboards with Marimo
- Create performance monitoring interfaces
- Debug multi-step reasoning pipelines

---

### Module 08: Building Custom DSPy Modules & Components

**Duration:** 75-90 minutes  
**Description:** Learn to create custom DSPy modules, extend functionality, and build reusable components

**Learning Goals:**

- Design and implement custom DSPy modules
- Extend DSPy functionality for specific use cases
- Build reusable component libraries
- Create module testing frameworks

**Key Concepts:** Custom module design, DSPy architecture, Component reusability, Module testing, Interactive development

**Practical Exercises:**

- Build custom reasoning modules
- Create domain-specific DSPy components
- Implement module testing suites
- Design interactive module builders

---

### Module 09: Production Deployment & Scaling

**Duration:** 60-75 minutes  
**Description:** Learn to deploy DSPy applications to production with proper scaling, monitoring, and maintenance

**Learning Goals:**

- Deploy DSPy applications to production
- Implement scaling and optimization strategies
- Set up monitoring and alerting systems
- Design maintenance and update workflows

**Key Concepts:** Production deployment, Application scaling, Monitoring systems, Maintenance workflows, Performance optimization

**Practical Exercises:**

- Deploy DSPy applications with containerization
- Implement monitoring and alerting
- Build deployment automation scripts
- Create maintenance dashboards

---

### Module 10: Advanced Projects & Case Studies

**Duration:** 120-150 minutes  
**Description:** Apply all learned concepts to build sophisticated agentic systems and complex AI applications

**Learning Goals:**

- Build end-to-end agentic systems
- Implement complex multi-agent workflows
- Create sophisticated RAG applications
- Design complete AI-powered solutions

**Key Concepts:** Advanced system design, Multi-agent systems, Complex workflows, End-to-end solutions, Real-world applications

**Practical Exercises:**

- Build a multi-agent research assistant
- Create an intelligent document processing system
- Implement a code analysis and generation tool
- Design a conversational AI platform

---

## Repository Folder Structure

```bash
DSPy-Zero-to-Expert/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
├── LICENSE
├── CONTRIBUTING.md
├── 00-setup/
│   ├── README.md
│   ├── setup_environment.py
│   ├── test_installation.py
│   ├── hello_dspy_marimo.py
│   ├── assets/
│   └── exercises/
├── 01-foundations/
│   ├── README.md
│   ├── signatures_basics.py
│   ├── module_comparison.py
│   ├── interactive_signature_tester.py
│   ├── exercises/
│   └── solutions/
├── 02-advanced-modules/
│   ├── README.md
│   ├── react_implementation.py
│   ├── tool_integration.py
│   ├── multi_step_reasoning.py
│   ├── debugging_dashboard.py
│   ├── exercises/
│   └── solutions/
├── 03-retrieval-rag/
│   ├── README.md
│   ├── rag_implementation.py
│   ├── vector_database_setup.py
│   ├── retrieval_optimization.py
│   ├── rag_evaluation_interface.py
│   ├── exercises/
│   ├── solutions/
│   └── data/
├── 04-optimization-teleprompters/
│   ├── README.md
│   ├── bootstrap_optimization.py
│   ├── mipro_implementation.py
│   ├── custom_metrics.py
│   ├── optimization_dashboard.py
│   ├── exercises/
│   └── solutions/
├── 05-evaluation-metrics/
│   ├── README.md
│   ├── evaluation_framework.py
│   ├── custom_metrics_library.py
│   ├── evaluation_dashboard.py
│   ├── ab_testing_framework.py
│   ├── exercises/
│   └── solutions/
├── 06-datasets-examples/
│   ├── README.md
│   ├── dataset_management.py
│   ├── data_preprocessing.py
│   ├── dataset_explorer.py
│   ├── data_quality_metrics.py
│   ├── exercises/
│   ├── solutions/
│   └── sample_data/
├── 07-tracing-debugging/
│   ├── README.md
│   ├── tracing_implementation.py
│   ├── debugging_utilities.py
│   ├── observability_dashboard.py
│   ├── performance_monitor.py
│   ├── exercises/
│   └── solutions/
├── 08-custom-modules/
│   ├── README.md
│   ├── custom_module_template.py
│   ├── component_library.py
│   ├── module_testing_framework.py
│   ├── interactive_builder.py
│   ├── exercises/
│   ├── solutions/
│   └── examples/
├── 09-production-deployment/
│   ├── README.md
│   ├── deployment_guide.py
│   ├── monitoring_setup.py
│   ├── scaling_strategies.py
│   ├── maintenance_dashboard.py
│   ├── exercises/
│   ├── solutions/
│   ├── deployment/
│   └── docker/
├── 10-advanced-projects/
│   ├── README.md
│   ├── multi_agent_system.py
│   ├── document_processing_system.py
│   ├── code_analysis_tool.py
│   ├── conversational_ai_platform.py
│   ├── project_templates/
│   ├── solutions/
│   └── case_studies/
├── common/
│   ├── __init__.py
│   ├── utils.py
│   ├── config.py
│   ├── marimo_components.py
│   ├── dspy_extensions.py
│   └── evaluation_utils.py
├── assets/
│   ├── images/
│   ├── diagrams/
│   ├── data_samples/
│   └── templates/
└── docs/
    ├── API_REFERENCE.md
    ├── TROUBLESHOOTING.md
    ├── BEST_PRACTICES.md
    └── ADVANCED_TOPICS.md
```

## Best Practices for DSPy + Marimo Integration

### 1. Marimo Notebook Structure

Each lesson should follow this Marimo cell structure:

```python
import marimo as mo
import dspy
import os
from typing import List, Dict, Any

# Configuration cell
mo.md("# Module Title: Learning Objectives")

# Setup cell
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Interactive parameter cell
parameters = mo.ui.dictionary({
    "model": mo.ui.dropdown(["gpt-4o-mini", "gpt-4o", "claude-3-sonnet"], value="gpt-4o-mini"),
    "provider": mo.ui.dropdown(["openai", "anthropic"], value="openai")
})

# Implementation cell  
def create_signature_demo():
    # DSPy implementation here
    pass

# Results visualization cell
results = mo.ui.table(data, selection="multi")

# Interactive analysis cell
analysis = mo.ui.text_area(placeholder="Enter your analysis...")
```

### 2. Progressive Complexity

- **Start Simple:** Each module begins with basic concepts
- **Build Incrementally:** Each lesson builds on previous knowledge
- **Interactive Discovery:** Use Marimo's reactivity to explore concepts
- **Practical Application:** Always include hands-on exercises

### 3. Integration Patterns

**DSPy Signatures + Marimo UI:**

```python
# Create interactive signature testing
signature_input = mo.ui.text_area(placeholder="Enter your signature...")
signature_test = dspy.Predict(signature_input.value)
result = signature_test(input_data)
mo.ui.table(result)
```

**Optimization Visualization:**

```python
# Track optimization progress with Marimo charts
optimization_progress = mo.ui.plotly(optimization_data)
metric_trends = mo.ui.altair_chart(metrics_df)
```

### 4. Common Utilities Structure

The `common/` directory should contain:

- **marimo_components.py:** Reusable Marimo UI components
- **dspy_extensions.py:** Custom DSPy modules and utilities
- **evaluation_utils.py:** Evaluation and metrics utilities
- **config.py:** Configuration management
- **utils.py:** General utility functions

## Getting Started

1. **Fork/Clone the Repository**
2. **Set up Environment** (follow setup instructions above)
3. **Start with Module 00** for environment setup
4. **Progress Sequentially** through the modules
5. **Complete Practical Exercises** in each module
6. **Build Your Final Project** in Module 10

## Recommended Learning Path

- **Beginner Path:** Follow modules 00-06 sequentially
- **Intermediate Path:** Skip to modules 02-08 if familiar with basics
- **Advanced Path:** Focus on modules 04, 07-10 for optimization and production
- **Project-Based:** Jump to Module 10 and reference earlier modules as needed

## Community & Support

- **GitHub Issues:** Report bugs or request features
- **Discussion Forum:** Ask questions and share insights
- **Contributing:** Submit improvements and additional content
- **Showcase:** Share your projects built with DSPy + Marimo

---

This learning repository provides a comprehensive, hands-on approach to mastering DSPy through interactive Marimo notebooks, designed specifically for experienced Python developers looking to build sophisticated AI applications.
