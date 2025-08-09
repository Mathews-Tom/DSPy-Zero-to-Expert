# DSPy Zero-to-Expert: Interactive Learning Repository

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-2.5+-green.svg)](https://dspy.ai/)
[![Marimo](https://img.shields.io/badge/Marimo-0.9+-purple.svg)](https://marimo.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**From Prompting to Programming: Code Your Way to Declarative AI Mastery**

A comprehensive, interactive learning course that teaches DSPy concepts through progressive, hands-on lessons using Marimo reactive notebooks. Designed for expert Python developers with GenAI and agentic system experience.

**DSPy (Declarative Self-improving Python)** is a declarative framework for building modular AI software. It allows you to **iterate fast on structured code**, rather than brittle strings, and offers algorithms that **compile AI programs into effective prompts and weights** for your language models, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

Instead of wrangling prompts or training jobs, **DSPy (Declarative Self-improving Python)** enables you to **build AI software from natural-language modules** and to generically compose them with different models, inference strategies, or learning algorithms. This makes AI software **more reliable, maintainable, and portable** across models and strategies.

## 🎯 Course Overview

**Target Audience:** Expert Python developers with GenAI and Agentic system experience  
**Duration:** 10-15 hours of interactive learning  
**Format:** Progressive modules with hands-on exercises

### Prerequisites

- Advanced Python programming experience
- Knowledge of generative AI concepts and applications  
- Experience with agentic systems and LLM applications
- Familiarity with machine learning workflows
- Understanding of prompt engineering principles

### Learning Objectives

By the end of this course, you will be able to:

- ✅ Master all core DSPy components and modules
- ✅ Learn practical Marimo usage for interactive AI development
- ✅ Build advanced RAG and multi-step reasoning systems
- ✅ Understand DSPy optimization and evaluation techniques
- ✅ Create production-ready agentic applications

## 🚀 Quick Start

### 1. Install uv Package Manager

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip/pipx
pip install uv
```

### 2. Clone and Setup

```bash
# Clone the repository
gh repo clone Mathews-Tom/DSPy-Zero-to-Expert
cd dspy-zero-to-expert

# Install dependencies
uv sync
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your API keys
# At minimum, configure OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 4. Verify Installation

```bash
uv run verify_installation.py
```

### 5. Start Learning

```bash
# Begin with Module 00
uv run marimo run 00-setup/hello_dspy_marimo.py
```

## 📚 Course Modules

| Module | Topic | Duration | Description |
|--------|-------|----------|-------------|
| **00** | [Environment Setup & Introduction](00-setup/) | 45-60 min | Set up development environment and create your first DSPy + Marimo application |
| **01** | [DSPy Foundations](01-foundations/) | 60-75 min | Master Signatures and basic modules with interactive testing |
| **02** | [Advanced Modules](02-advanced-modules/) | 75-90 min | ReAct agents, tool integration, and multi-step reasoning |
| **03** | [Retrieval-Augmented Generation](03-retrieval-rag/) | 90-105 min | Complete RAG systems with vector databases and optimization |
| **04** | [Optimization & Teleprompters](04-optimization-teleprompters/) | 75-90 min | Automatic prompt optimization with BootstrapFewShot and MIPRO |
| **05** | [Evaluation & Metrics](05-evaluation-metrics/) | 60-75 min | Design evaluation frameworks and custom metrics |
| **06** | [Datasets & Examples](06-datasets-examples/) | 45-60 min | Work with datasets and manage training data |
| **07** | [Tracing & Debugging](07-tracing-debugging/) | 60-75 min | Master debugging and observability tools |
| **08** | [Custom Modules](08-custom-modules/) | 75-90 min | Build custom DSPy modules and reusable components |
| **09** | [Production Deployment](09-production-deployment/) | 60-75 min | Deploy and scale DSPy applications |
| **10** | [Advanced Projects](10-advanced-projects/) | 120-150 min | Build sophisticated agentic systems and AI applications |

## 🛠️ Technology Stack

- **Core Framework**: [DSPy](https://dspy.ai/) for language model programming
- **Interactive Environment**: [Marimo](https://marimo.io/) for reactive notebooks
- **Package Management**: [uv](https://github.com/astral-sh/uv) for fast dependency resolution
- **Language Models**: OpenAI, Anthropic, Cohere integration
- **Vector Databases**: FAISS, ChromaDB, Qdrant support
- **Evaluation**: Custom metrics and A/B testing frameworks
- **Visualization**: Plotly, Matplotlib for interactive charts

## 🎓 Learning Paths

### Beginner Path

Follow modules 00-06 sequentially for a complete foundation.

### Intermediate Path  

Skip to modules 02-08 if familiar with DSPy basics.

### Advanced Path

Focus on modules 04, 07-10 for optimization and production.

### Project-Based

Jump to Module 10 and reference earlier modules as needed.

## 📖 Key Features

### 🔄 Reactive Learning

- **Interactive Notebooks**: Marimo's reactive programming model
- **Real-time Updates**: Parameter changes immediately update results
- **Visual Feedback**: Interactive charts and visualizations

### 🧠 Comprehensive DSPy Coverage

- **All Core Modules**: Predict, ChainOfThought, ReAct, and more
- **Advanced Patterns**: Multi-agent systems, tool integration
- **Optimization**: BootstrapFewShot, MIPRO, custom metrics
- **Production Ready**: Deployment, scaling, monitoring

### 🛠️ Hands-on Exercises

- **Progressive Complexity**: Each module builds on previous knowledge
- **Practical Applications**: Real-world use cases and examples
- **Solution Validation**: Automated checking and feedback

## 📁 Repository Structure

```bash
DSPy-Zero-to-Expert/
├── 00-setup/                       # Environment setup and introduction
├── 01-foundations/                 # DSPy signatures and basic modules
├── 02-advanced-modules/            # ReAct, tools, multi-step reasoning
├── 03-retrieval-rag/               # RAG implementation and optimization
├── 04-optimization-teleprompters/  # Automatic optimization
├── 05-evaluation-metrics/          # Evaluation frameworks and metrics
├── 06-datasets-examples/           # Dataset management and examples
├── 07-tracing-debugging/           # Debugging and observability
├── 08-custom-modules/              # Custom module development
├── 09-production-deployment/       # Production deployment and scaling
├── 10-advanced-projects/           # Advanced projects and case studies
├── common/                         # Shared utilities and components
├── assets/                         # Images, diagrams, and templates
├── docs/                           # Documentation and guides
├── pyproject.toml                  # Project configuration
├── .env.template                   # Environment configuration template
└── verify_installation.py          # Installation verification script
``` 

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check --fix .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Best Practices**: [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)
- **Issues**: [GitHub Issues](https://github.com/Mathews-Tom/DSPy-Zero-to-Expert/issues)

## 🙏 Acknowledgments

- [DSPy Team](https://dspy.ai/) for the amazing framework
- [Marimo Team](https://marimo.io/) for the reactive notebook environment
- [uv Team](https://github.com/astral-sh/uv) for fast Python package management

---

**Ready to become a DSPy expert?** Start with [Module 00: Environment Setup](00-setup/) and begin your journey! 🚀
