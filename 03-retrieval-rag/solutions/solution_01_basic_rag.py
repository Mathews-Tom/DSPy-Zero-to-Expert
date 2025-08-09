# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Dict, List

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return (
        Any,
        Dict,
        List,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🎯 Solution 01: Basic RAG Implementation

            **Complete solution** for building a RAG system from scratch using DSPy modules.

            ## 📋 Solution Overview

            This solution demonstrates:  
            1. **Simple Document Store** - In-memory storage with keyword-based retrieval  
            2. **RAG Signature Design** - Well-structured signature for context-based QA  
            3. **Complete RAG Module** - Full pipeline with error handling  
            4. **Comprehensive Testing** - Multiple test cases and edge case handling  

            ## 🏗️ Architecture

            **Components:**  
            - `SimpleDocumentStore` - Document storage and retrieval  
            - `BasicRAGSignature` - DSPy signature for RAG  
            - `BasicRAGModule` - Complete RAG implementation  
            - `RAGTester` - Testing and validation framework  

            Let's build it step by step! 🚀
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

                Ready to implement the complete RAG solution!
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
def _(Any, Dict, List, available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Part A: Document Storage System

                **Complete implementation** of a simple document store with keyword-based retrieval:
                """
            )
        )

        class SimpleDocumentStore:
            """Simple in-memory document store with keyword-based retrieval."""

            def __init__(self):
                self.documents = {}
                self.metadata = {}
                self.doc_count = 0

            def add_document(
                self, doc_id: str, text: str, metadata: Dict[str, Any] = None
            ):
                """Add a document to the store."""
                if not doc_id or not text:
                    raise ValueError("Document ID and text cannot be empty")

                self.documents[doc_id] = (
                    text.lower()
                )  # Store lowercase for better matching
                self.metadata[doc_id] = metadata or {}
                self.doc_count += 1

                return True

            def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
                """Search for relevant documents using keyword matching."""
                if not query:
                    return []

                query_words = set(query.lower().split())
                scored_docs = []

                for doc_id, doc_text in self.documents.items():
                    doc_words = set(doc_text.split())

                    # Calculate simple keyword overlap score
                    overlap = len(query_words.intersection(doc_words))
                    if overlap > 0:
                        # Normalize by query length for better scoring
                        score = overlap / len(query_words)
                        scored_docs.append(
                            {
                                "id": doc_id,
                                "text": self.documents[doc_id],
                                "metadata": self.metadata[doc_id],
                                "score": score,
                            }
                        )

                # Sort by score and return top_k
                scored_docs.sort(key=lambda x: x["score"], reverse=True)
                return scored_docs[:top_k]

            def get_stats(self) -> Dict[str, Any]:
                """Get document store statistics."""
                return {
                    "total_documents": self.doc_count,
                    "average_doc_length": sum(
                        len(doc.split()) for doc in self.documents.values()
                    )
                    / max(self.doc_count, 1),
                }

        # Create sample document store with test data
        sample_doc_store = SimpleDocumentStore()

        # Add sample documents about Python programming
        sample_documents = [
            (
                "python_basics",
                "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                {"topic": "basics", "difficulty": "beginner"},
            ),
            (
                "python_data_types",
                "Python has several built-in data types including integers, floats, strings, lists, tuples, dictionaries, and sets. Each data type has specific methods and properties.",
                {"topic": "data_types", "difficulty": "beginner"},
            ),
            (
                "python_functions",
                "Functions in Python are defined using the def keyword. They can accept parameters, return values, and support default arguments, keyword arguments, and variable-length arguments.",
                {"topic": "functions", "difficulty": "intermediate"},
            ),
            (
                "python_classes",
                "Object-oriented programming in Python uses classes to create objects. Classes define attributes and methods that objects can use. Python supports inheritance, encapsulation, and polymorphism.",
                {"topic": "oop", "difficulty": "intermediate"},
            ),
            (
                "python_modules",
                "Python modules are files containing Python code that can be imported and used in other programs. The import statement is used to include modules. Python has a rich standard library.",
                {"topic": "modules", "difficulty": "intermediate"},
            ),
            (
                "python_exceptions",
                "Exception handling in Python uses try, except, else, and finally blocks. Common exceptions include ValueError, TypeError, and IndexError. Custom exceptions can be created by inheriting from Exception.",
                {"topic": "exceptions", "difficulty": "intermediate"},
            ),
            (
                "python_file_io",
                "File input/output in Python uses the open() function. Files can be opened in different modes like read, write, or append. Context managers with 'with' statements ensure proper file handling.",
                {"topic": "file_io", "difficulty": "intermediate"},
            ),
            (
                "python_libraries",
                "Python has extensive third-party libraries available through pip. Popular libraries include NumPy for numerical computing, Pandas for data analysis, and Requests for HTTP requests.",
                {"topic": "libraries", "difficulty": "advanced"},
            ),
            (
                "python_web_frameworks",
                "Python web frameworks like Django and Flask enable web application development. Django is a full-featured framework while Flask is lightweight and flexible.",
                {"topic": "web", "difficulty": "advanced"},
            ),
            (
                "python_machine_learning",
                "Python is widely used in machine learning with libraries like scikit-learn, TensorFlow, and PyTorch. These libraries provide tools for data preprocessing, model training, and evaluation.",
                {"topic": "ml", "difficulty": "advanced"},
            ),
        ]

        for doc_id, text, metadata in sample_documents:
            sample_doc_store.add_document(doc_id, text, metadata)

        cell3_content = mo.md(
            cleandoc(
                f"""
                ### 🏗️ Document Store Implementation Complete

                **Features:**  
                - **Keyword-based retrieval** with overlap scoring  
                - **Metadata support** for additional document information  
                - **Error handling** for invalid inputs  
                - **Statistics tracking** for monitoring  

                **Sample Data Loaded:**  
                - **{sample_doc_store.get_stats()['total_documents']} documents** about Python programming  
                - **Average document length:** {sample_doc_store.get_stats()['average_doc_length']:.1f} words  
                - **Topics covered:** Basics, data types, functions, OOP, modules, exceptions, file I/O, libraries, web frameworks, machine learning  

                The document store is ready for RAG integration!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        SimpleDocumentStore = None
        sample_doc_store = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return SimpleDocumentStore, sample_doc_store


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Part B: RAG Signature Design

                **Well-structured signature** for context-based question answering:
                """
            )
        )

        class BasicRAGSignature(dspy.Signature):
            """Answer questions based on retrieved document context.

            Use the provided context to answer the user's question accurately and completely.
            If the context doesn't contain enough information, acknowledge the limitation.
            """

            question = dspy.InputField(desc="The user's question to be answered")
            context = dspy.InputField(
                desc="Retrieved document context relevant to the question"
            )

            answer = dspy.OutputField(
                desc="Comprehensive answer based on the provided context"
            )
            confidence = dspy.OutputField(
                desc="Confidence level in the answer (high/medium/low)"
            )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🎯 RAG Signature Design Complete

                **Signature Features:**  
                - **Clear purpose** - Focused on context-based question answering  
                - **Detailed descriptions** - Each field has clear guidance  
                - **Confidence scoring** - Helps assess answer quality  
                - **Limitation handling** - Instructs model to acknowledge when context is insufficient  

                **Input Fields:**  
                - `question` - The user's question  
                - `context` - Retrieved document context  

                **Output Fields:**  
                - `answer` - Comprehensive response  
                - `confidence` - Quality assessment  

                The signature provides clear guidance for the language model!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        BasicRAGSignature = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (BasicRAGSignature,)


@app.cell
def _(
    BasicRAGSignature,
    SimpleDocumentStore,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
    time,
):
    if available_providers and BasicRAGSignature and SimpleDocumentStore:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🔧 Part C: RAG Module Implementation

                **Complete RAG module** with retrieval, generation, and error handling:
                """
            )
        )

        class BasicRAGModule(dspy.Module):
            """Complete RAG module combining retrieval and generation."""

            def __init__(self, document_store: SimpleDocumentStore, top_k: int = 3):
                super().__init__()
                self.document_store = document_store
                self.top_k = top_k
                self.rag_predictor = dspy.ChainOfThought(BasicRAGSignature)

                # Track usage statistics
                self.query_count = 0
                self.total_retrieval_time = 0
                self.total_generation_time = 0

            def forward(self, question: str):
                """Execute the complete RAG pipeline."""
                if not question or not question.strip():
                    return dspy.Prediction(
                        answer="I need a question to answer. Please provide a specific question.",
                        confidence="low",
                        retrieved_docs=[],
                        error="Empty question provided",
                    )

                try:
                    # Step 1: Retrieve relevant documents
                    retrieval_start = time.time()
                    retrieved_docs = self.document_store.search(
                        question, top_k=self.top_k
                    )
                    retrieval_time = time.time() - retrieval_start

                    if not retrieved_docs:
                        return dspy.Prediction(
                            answer="I couldn't find any relevant information to answer your question. Please try rephrasing or asking about a different topic.",
                            confidence="low",
                            retrieved_docs=[],
                            retrieval_time=retrieval_time,
                        )

                    # Step 2: Prepare context from retrieved documents
                    context_parts = []
                    for i, doc in enumerate(retrieved_docs, 1):
                        context_parts.append(
                            f"Document {i} (relevance: {doc['score']:.2f}): {doc['text']}"
                        )

                    context = "\n\n".join(context_parts)

                    # Step 3: Generate answer using the RAG signature
                    generation_start = time.time()
                    result = self.rag_predictor(question=question, context=context)
                    generation_time = time.time() - generation_start

                    # Step 4: Update statistics
                    self.query_count += 1
                    self.total_retrieval_time += retrieval_time
                    self.total_generation_time += generation_time

                    return dspy.Prediction(
                        answer=result.answer,
                        confidence=result.confidence,
                        retrieved_docs=retrieved_docs,
                        context=context,
                        retrieval_time=retrieval_time,
                        generation_time=generation_time,
                        total_time=retrieval_time + generation_time,
                    )

                except Exception as e:
                    return dspy.Prediction(
                        answer=f"I encountered an error while processing your question: {str(e)}",
                        confidence="low",
                        retrieved_docs=[],
                        error=str(e),
                    )

            def get_stats(self):
                """Get RAG module usage statistics."""
                if self.query_count == 0:
                    return {"queries_processed": 0}

                return {
                    "queries_processed": self.query_count,
                    "avg_retrieval_time": self.total_retrieval_time / self.query_count,
                    "avg_generation_time": self.total_generation_time
                    / self.query_count,
                    "avg_total_time": (
                        self.total_retrieval_time + self.total_generation_time
                    )
                    / self.query_count,
                }

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🔧 RAG Module Implementation Complete

                **Module Features:**  
                - **Complete pipeline** - Retrieval → Context preparation → Generation  
                - **Error handling** - Graceful handling of empty queries, no results, and exceptions  
                - **Performance tracking** - Timing statistics for optimization  
                - **Flexible configuration** - Configurable top_k for retrieval  
                - **Rich output** - Includes answer, confidence, retrieved docs, and timing info  

                **Pipeline Steps:**  
                1. **Input validation** - Check for empty or invalid questions  
                2. **Document retrieval** - Search document store for relevant content  
                3. **Context preparation** - Format retrieved documents for the model  
                4. **Answer generation** - Use DSPy ChainOfThought for reasoning  
                5. **Statistics tracking** - Monitor performance metrics  

                The RAG module is ready for comprehensive testing!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        BasicRAGModule = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (BasicRAGModule,)


@app.cell
def _(
    BasicRAGModule,
    available_providers,
    cleandoc,
    mo,
    output,
    sample_doc_store,
):
    if available_providers and BasicRAGModule and sample_doc_store:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🧪 Part D: Testing and Validation

                **Comprehensive testing framework** with multiple test cases and edge cases:
                """
            )
        )

        class RAGTester:
            """Comprehensive testing framework for RAG systems."""

            def __init__(self, rag_module: BasicRAGModule):
                self.rag_module = rag_module
                self.test_results = []

            def run_test_case(
                self, question: str, expected_topics: list = None, test_name: str = ""
            ):
                """Run a single test case and evaluate results."""
                print(f"\n{'='*50}")
                print(f"TEST: {test_name or question[:50]}...")
                print(f"{'='*50}")

                result = self.rag_module(question)

                # Display results
                print(f"Question: {question}")
                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence}")

                if hasattr(result, "retrieved_docs") and result.retrieved_docs:
                    print(f"Retrieved {len(result.retrieved_docs)} documents:")
                    for i, doc in enumerate(result.retrieved_docs, 1):
                        print(f"  {i}. {doc['id']} (score: {doc['score']:.3f})")

                if hasattr(result, "total_time"):
                    print(f"Total time: {result.total_time:.3f}s")

                # Evaluate quality (simple heuristics)
                quality_score = self._evaluate_quality(result, expected_topics)
                print(f"Quality score: {quality_score:.2f}/5.0")

                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "quality_score": quality_score,
                    "retrieved_count": (
                        len(result.retrieved_docs)
                        if hasattr(result, "retrieved_docs") and result.retrieved_docs
                        else 0
                    ),
                    "has_error": hasattr(result, "error"),
                }

                self.test_results.append(test_result)
                return result

            def _evaluate_quality(self, result, expected_topics=None):
                """Simple quality evaluation based on heuristics."""
                score = 0.0

                # Check if answer exists and is substantial
                if (
                    hasattr(result, "answer")
                    and result.answer
                    and len(result.answer) > 20
                ):
                    score += 1.0

                # Check confidence level
                if hasattr(result, "confidence"):
                    if result.confidence.lower() == "high":
                        score += 1.5
                    elif result.confidence.lower() == "medium":
                        score += 1.0
                    else:
                        score += 0.5

                # Check if documents were retrieved
                if hasattr(result, "retrieved_docs") and result.retrieved_docs:
                    score += 1.0

                # Check for expected topics (if provided)
                if expected_topics and hasattr(result, "answer"):
                    topic_matches = sum(
                        1
                        for topic in expected_topics
                        if topic.lower() in result.answer.lower()
                    )
                    score += min(topic_matches * 0.5, 1.5)

                return min(score, 5.0)

            def run_comprehensive_test_suite(self):
                """Run comprehensive test suite covering various scenarios."""
                print("🧪 Running Comprehensive RAG Test Suite")
                print("=" * 60)

                # Test Case 1: Basic factual question
                self.run_test_case(
                    "What is Python?",
                    expected_topics=["programming", "language"],
                    test_name="Basic Factual Question",
                )

                # Test Case 2: Specific technical question
                self.run_test_case(
                    "How do you define functions in Python?",
                    expected_topics=["def", "function", "parameters"],
                    test_name="Technical Question",
                )

                # Test Case 3: Comparison question
                self.run_test_case(
                    "What's the difference between lists and tuples in Python?",
                    expected_topics=["list", "tuple", "data type"],
                    test_name="Comparison Question",
                )

                # Test Case 4: Advanced topic
                self.run_test_case(
                    "What Python libraries are used for machine learning?",
                    expected_topics=["scikit-learn", "tensorflow", "pytorch"],
                    test_name="Advanced Topic",
                )

                # Test Case 5: Edge case - empty question
                self.run_test_case("", test_name="Edge Case: Empty Question")

                # Test Case 6: Edge case - nonsense question
                self.run_test_case(
                    "askdjfh askdjfh random nonsense",
                    test_name="Edge Case: Nonsense Question",
                )

                # Test Case 7: Question about unavailable topic
                self.run_test_case(
                    "How do you cook pasta?", test_name="Edge Case: Unavailable Topic"
                )

                # Generate summary report
                self._generate_test_report()

            def _generate_test_report(self):
                """Generate comprehensive test report."""
                print("\n" + "=" * 60)
                print("📊 TEST SUMMARY REPORT")
                print("=" * 60)

                if not self.test_results:
                    print("No test results available.")
                    return

                total_tests = len(self.test_results)
                successful_tests = sum(
                    1 for result in self.test_results if not result["has_error"]
                )
                avg_quality = (
                    sum(result["quality_score"] for result in self.test_results)
                    / total_tests
                )

                print(f"Total tests run: {total_tests}")
                print(f"Successful tests: {successful_tests}")
                print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
                print(f"Average quality score: {avg_quality:.2f}/5.0")

                # Quality distribution
                high_quality = sum(
                    1 for result in self.test_results if result["quality_score"] >= 4.0
                )
                medium_quality = sum(
                    1
                    for result in self.test_results
                    if 2.0 <= result["quality_score"] < 4.0
                )
                low_quality = sum(
                    1 for result in self.test_results if result["quality_score"] < 2.0
                )

                print("\nQuality Distribution:")
                print(f"  High quality (4.0+): {high_quality} tests")
                print(f"  Medium quality (2.0-3.9): {medium_quality} tests")
                print(f"  Low quality (<2.0): {low_quality} tests")

                # RAG module statistics
                rag_stats = self.rag_module.get_stats()
                if rag_stats.get("queries_processed", 0) > 0:
                    print("\nPerformance Statistics:")
                    print(
                        f"  Average retrieval time: {rag_stats['avg_retrieval_time']:.3f}s"
                    )
                    print(
                        f"  Average generation time: {rag_stats['avg_generation_time']:.3f}s"
                    )
                    print(f"  Average total time: {rag_stats['avg_total_time']:.3f}s")

        # Create RAG system and tester
        rag_system = BasicRAGModule(sample_doc_store, top_k=3)
        rag_tester = RAGTester(rag_system)

        cell6_content = mo.md(
            cleandoc(
                """
                ### 🧪 Testing Framework Complete

                **Testing Features:**  
                - **Comprehensive test cases** - Factual, technical, comparison, and advanced questions  
                - **Edge case handling** - Empty queries, nonsense input, unavailable topics  
                - **Quality evaluation** - Multi-factor scoring system  
                - **Performance monitoring** - Timing and efficiency metrics  
                - **Detailed reporting** - Summary statistics and quality distribution  

                **Test Categories:**  
                1. **Basic factual questions** - Simple information retrieval  
                2. **Technical questions** - Specific implementation details  
                3. **Comparison questions** - Multi-concept analysis  
                4. **Advanced topics** - Complex domain knowledge  
                5. **Edge cases** - Error handling and robustness  

                Ready to run the comprehensive test suite!
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        RAGTester = None
        rag_system = None
        rag_tester = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (rag_tester,)


@app.cell
def _(available_providers, mo, output, rag_tester):
    if available_providers and rag_tester:
        # Create test execution button
        cell7_test_button = mo.ui.run_button(label="🧪 Run Comprehensive Test Suite")

        cell7_out = mo.vstack(
            [
                mo.md("## 🚀 Execute Comprehensive Testing"),
                mo.md(
                    "Click the button below to run the complete test suite and see the RAG system in action:"
                ),
                cell7_test_button,
            ]
        )
    else:
        cell7_test_button = None
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return (cell7_test_button,)


@app.cell
def _(available_providers, cell7_test_button, mo, output, rag_tester):
    if (
        available_providers
        and cell7_test_button
        and cell7_test_button.value
        and rag_tester
    ):
        # Run the comprehensive test suite
        rag_tester.run_comprehensive_test_suite()

        cell8_out = mo.md(
            """
            ## ✅ Test Suite Execution Complete!

            The comprehensive test suite has been executed. Check the output above for detailed results including:

            - Individual test case results with answers and quality scores  
            - Performance metrics and timing information  
            - Summary report with success rates and quality distribution  
            - RAG system statistics and performance analysis  

            **Key Insights:**  
            - The system handles various question types effectively  
            - Edge cases are managed gracefully with appropriate error messages  
            - Performance metrics help identify optimization opportunities  
            - Quality scoring provides objective evaluation criteria  
            """
        )
    else:
        cell8_out = mo.md(
            "*Click the test button above to run the comprehensive test suite*"
        )

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_out = mo.md(
            cleandoc(
                """
                ## 🎉 Solution Complete: Basic RAG Implementation

                ### 🏆 What We've Built

                **Complete RAG System Components:**  
                - ✅ **Document Store** - Simple but effective keyword-based retrieval  
                - ✅ **RAG Signature** - Well-designed signature with confidence scoring  
                - ✅ **RAG Module** - Full pipeline with error handling and performance tracking  
                - ✅ **Testing Framework** - Comprehensive evaluation with quality metrics  

                ### 🔍 Key Implementation Insights

                **Design Decisions:**  
                - **Simple keyword matching** - Easy to understand and debug, good baseline  
                - **Confidence scoring** - Helps assess answer quality and reliability  
                - **Error handling** - Graceful degradation for edge cases  
                - **Performance tracking** - Essential for optimization and monitoring  

                **Quality Factors:**  
                - **Answer completeness** - Substantial, informative responses  
                - **Context utilization** - Effective use of retrieved documents  
                - **Confidence calibration** - Appropriate confidence levels  
                - **Error resilience** - Robust handling of problematic inputs  

                ### 🚀 Next Steps and Improvements

                **Immediate Enhancements:**  
                1. **Better retrieval** - TF-IDF, BM25, or semantic similarity  
                2. **Query expansion** - Synonyms and related terms  
                3. **Answer validation** - Fact-checking and consistency  
                4. **Caching** - Store frequent queries for faster response  

                **Advanced Features:**
                1. **Multi-document synthesis** - Combine information from multiple sources  
                2. **Relevance thresholding** - Filter out low-quality matches  
                3. **Dynamic top-k** - Adjust retrieval count based on query complexity  
                4. **Learning from feedback** - Improve based on user interactions  

                ### 💡 Production Considerations

                **Scalability:**  
                - Replace in-memory storage with persistent database  
                - Implement proper indexing for large document collections  
                - Add caching layers for frequently accessed content  

                **Quality Assurance:**  
                - Implement comprehensive evaluation metrics  
                - Add human feedback collection and analysis  
                - Create A/B testing framework for improvements  

                **Monitoring:**  
                - Track query patterns and performance metrics  
                - Monitor answer quality and user satisfaction  
                - Alert on performance degradation or errors  

                ### 🎯 Learning Outcomes

                You've successfully built a complete RAG system that demonstrates:  
                - **End-to-end pipeline** from question to answer  
                - **Proper error handling** and edge case management  
                - **Performance monitoring** and quality assessment  
                - **Modular design** that's easy to extend and improve  

                **Ready for the next challenge?** Try Exercise 02 to build advanced vector database systems! 🚀
                """
            )
        )
    else:
        cell9_out = mo.md("")

    output.replace(cell9_out)
    return


if __name__ == "__main__":
    app.run()
