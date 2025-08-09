# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return (
        cleandoc,
        get_config,
        mo,
        output,
        setup_dspy_environment,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📝 Exercise 02: Vector Database Integration

            **Objective:** Build a comprehensive vector database system with multiple backend support.

            ## 🎯 Your Mission

            Create a vector database system that:  
            1. Supports multiple vector database backends (simulated)  
            2. Implements proper vector similarity search  
            3. Handles metadata filtering and complex queries  
            4. Provides performance monitoring and optimization  

            ## 📋 Requirements

            **Part A: Vector Database Interface**  
            - Design a unified interface for vector databases  
            - Support basic operations: add, search, delete, update  
            - Include proper error handling and validation  
            - Add support for metadata filtering

            **Part B: Multiple Backend Implementation**  
            - Implement at least 2 different vector database backends (simulated)  
            - Each should have different characteristics (speed vs accuracy trade-offs)  
            - Include proper initialization and configuration  
            - Add connection management and cleanup  

            **Part C: Advanced Search Features**  
            - Implement similarity search with configurable distance metrics  
            - Add support for hybrid search (vector + metadata filtering)  
            - Include query expansion and refinement capabilities  
            - Add result ranking and re-ranking features  

            **Part D: Performance Monitoring**  
            - Track query performance and response times  
            - Monitor database size and memory usage  
            - Implement caching for frequently accessed vectors  
            - Add performance comparison between backends  

            ## 🚀 Bonus Challenges

            1. **Batch Operations:** Implement efficient batch insert/update operations  
            2. **Index Optimization:** Add different indexing strategies for performance  
            3. **Distributed Search:** Simulate distributed vector search across multiple nodes  
            4. **Vector Compression:** Implement vector compression techniques  
            5. **Real-time Updates:** Handle real-time vector updates without full reindexing  

            ## 💡 Hints

            - Use cosine similarity as your primary distance metric  
            - Implement proper vector normalization  
            - Consider memory vs speed trade-offs in your design  
            - Use mock/simulated implementations for complex database features  
            - Focus on clean interfaces that could easily be extended to real databases  

            Ready to build a production-ready vector database system? Let's dive in! 🚀
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(cleandoc, get_config, mo, output, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  

                Ready to build your vector database system!

                ## 📝 Implementation Guidelines

                - Start with the interface design - this will guide your implementation
                - Focus on clean, extensible code that could work with real databases
                - Test thoroughly with different vector sizes and query types
                - Consider edge cases like empty databases, duplicate vectors, etc.
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first to configure your API keys.
                """
            )
        )

    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🏗️ Part A: Vector Database Interface
    
                **TODO:** Design your unified vector database interface.
    
                ```python
                # Your interface implementation here
                from abc import ABC, abstractmethod
    
                class VectorDatabaseInterface(ABC):
                    \"\"\"Unified interface for vector databases\"\"\"
    
                    @abstractmethod
                    def initialize(self, config: dict) -> bool:
                        \"\"\"Initialize the database connection\"\"\"
                        pass
    
                    @abstractmethod
                    def add_vectors(self, vectors: list, metadata: list, ids: list = None) -> bool:
                        \"\"\"Add vectors with metadata\"\"\"
                        pass
    
                    @abstractmethod
                    def search(self, query_vector: list, top_k: int = 10, filters: dict = None) -> list:
                        \"\"\"Search for similar vectors\"\"\"
                        pass
    
                    # Add more methods as needed
                ```
                """
            )
        )
    else:
        cell3_out = mo.md("")

    output.replace(cell3_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell4_out = mo.md(
            cleandoc(
                """
                ## 🔧 Part B: Backend Implementations
    
                **TODO:** Implement at least 2 different vector database backends.
    
                ```python
                # Your backend implementations here
                class FastVectorDB(VectorDatabaseInterface):
                    \"\"\"Fast but less accurate vector database\"\"\"
    
                    def __init__(self):
                        # Initialize fast backend
                        pass
    
                    # Implement all interface methods
    
                class AccurateVectorDB(VectorDatabaseInterface):
                    \"\"\"Slower but more accurate vector database\"\"\"
    
                    def __init__(self):
                        # Initialize accurate backend
                        pass
    
                    # Implement all interface methods
                ```
                """
            )
        )
    else:
        cell4_out = mo.md("")

    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_out = mo.md(
            cleandoc(
                """
                ## 🎯 Part C: Advanced Search Features
    
                **TODO:** Implement advanced search capabilities.
    
                ```python
                # Your advanced search implementation here
                class AdvancedSearchEngine:
                    def __init__(self, vector_db):
                        self.vector_db = vector_db
    
                    def hybrid_search(self, query_vector, metadata_filters, top_k=10):
                        \"\"\"Combine vector similarity with metadata filtering\"\"\"
                        pass
    
                    def expand_query(self, original_query):
                        \"\"\"Expand query with related terms\"\"\"
                        pass
    
                    def rerank_results(self, results, query_context):
                        \"\"\"Re-rank results based on additional context\"\"\"
                        pass
                ```
                """
            )
        )
    else:
        cell5_out = mo.md("")

    output.replace(cell5_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell6_out = mo.md(
            cleandoc(
                """
                ## 📊 Part D: Performance Monitoring
    
                **TODO:** Implement performance monitoring and comparison.
    
                ```python
                # Your performance monitoring implementation here
                class VectorDBPerformanceMonitor:
                    def __init__(self):
                        self.metrics = {}
    
                    def track_query(self, db_name, query_time, result_count):
                        \"\"\"Track query performance\"\"\"
                        pass
    
                    def compare_backends(self, test_queries):
                        \"\"\"Compare performance across different backends\"\"\"
                        pass
    
                    def generate_report(self):
                        \"\"\"Generate performance report\"\"\"
                        pass
                ```
                """
            )
        )
    else:
        cell6_out = mo.md("")

    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_out = mo.md(
            cleandoc(
                """
                ## 🧪 Testing and Validation
    
                **TODO:** Create comprehensive tests for your vector database system.
    
                ```python
                # Your testing implementation here
                def test_vector_database_system():
                    \"\"\"Comprehensive test suite for vector database system\"\"\"
    
                    # Test basic operations
                    def test_basic_operations():
                        pass
    
                    # Test search accuracy
                    def test_search_accuracy():
                        pass
    
                    # Test performance
                    def test_performance():
                        pass
    
                    # Test edge cases
                    def test_edge_cases():
                        pass
    
                    # Run all tests
                    test_basic_operations()
                    test_search_accuracy()
                    test_performance()
                    test_edge_cases()
                ```
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell8_out = mo.md(
            cleandoc(
                """
                ## 🎉 Reflection and Next Steps
    
                **TODO:** After completing your implementation, consider:
    
                1. **Architecture Decisions:** What design choices did you make and why?
    
                2. **Performance Trade-offs:** How did different backends perform? What trade-offs did you observe?
    
                3. **Scalability Considerations:** How would your system handle millions of vectors?
    
                4. **Real-world Integration:** What would you need to change to use real vector databases?
    
                **Next Steps:**  
                - Experiment with different distance metrics  
                - Try the bonus challenges for advanced features  
                - Consider how this integrates with your RAG system from Exercise 01  
                - Think about production deployment considerations  
                """
            )
        )
    else:
        cell8_out = mo.md("")

    output.replace(cell8_out)
    return


if __name__ == "__main__":
    app.run()
