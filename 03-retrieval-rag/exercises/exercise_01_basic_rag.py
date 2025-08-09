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

    return cleandoc, get_config, mo, output, setup_dspy_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📝 Exercise 01: Basic RAG Implementation

            **Objective:** Build a complete RAG system from scratch using DSPy modules.

            ## 🎯 Your Mission

            Implement a basic RAG system that can:  
            1. Store and retrieve documents from a simple vector database  
            2. Generate answers based on retrieved context  
            3. Handle queries with proper error handling  

            ## 📋 Requirements

            **Part A: Document Storage System**  
            - Create a simple in-memory document store  
            - Implement basic text similarity search (you can use simple keyword matching)  
            - Store at least 10 sample documents about a topic of your choice  

            **Part B: RAG Signature Design**  
            - Design a signature for context-based question answering  
            - Include proper input/output field descriptions  
            - Consider what information the model needs to generate good answers  

            **Part C: RAG Module Implementation**  
            - Build a complete RAG module that combines retrieval and generation  
            - Implement proper error handling for missing documents  
            - Add logging or debugging output to track the process  

            **Part D: Testing and Validation**  
            - Create at least 5 test queries  
            - Test edge cases (no relevant documents, empty queries, etc.)  
            - Evaluate the quality of generated answers  

            ## 🚀 Bonus Challenges

            1. **Relevance Scoring:** Add a relevance threshold to filter out poor matches  
            2. **Multi-Document Synthesis:** Combine information from multiple retrieved documents  
            3. **Query Expansion:** Expand user queries with synonyms or related terms  
            4. **Answer Confidence:** Add confidence scoring to generated answers  

            ## 💡 Hints

            - Start simple with keyword-based retrieval before moving to more complex methods  
            - Use clear, descriptive field names in your signatures  
            - Test with both simple and complex queries  
            - Consider how to handle cases where no relevant documents are found  

            Ready to build your first RAG system? Let's get started! 🚀
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

                You're ready to start building your RAG system!

                ## 📝 Implementation Area

                Use the cells below to implement your solution. Remember to:  
                - Follow the requirements step by step  
                - Test your implementation thoroughly  
                - Add comments to explain your approach  
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
                ## 🏗️ Part A: Document Storage System
    
                **TODO:** Implement your document storage system here.
    
                ```python
                # Your implementation here
                class SimpleDocumentStore:
                    def __init__(self):
                        # Initialize your document store
                        pass
    
                    def add_document(self, doc_id, text, metadata=None):
                        # Add a document to the store
                        pass
    
                    def search(self, query, top_k=5):
                        # Search for relevant documents
                        pass
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
                ## 🎯 Part B: RAG Signature Design
    
                **TODO:** Design your RAG signature here.
    
                ```python
                # Your signature implementation here
                class YourRAGSignature(dspy.Signature):
                    \"\"\"Your signature description here\"\"\"
    
                    # Define your input and output fields
                    pass
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
                ## 🔧 Part C: RAG Module Implementation
    
                **TODO:** Build your complete RAG module here.
    
                ```python
                # Your RAG module implementation here
                class YourRAGModule(dspy.Module):
                    def __init__(self, document_store):
                        # Initialize your RAG module
                        pass
    
                    def forward(self, question):
                        # Implement the RAG pipeline
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
                ## 🧪 Part D: Testing and Validation
    
                **TODO:** Create your test cases and validation here.
    
                ```python
                # Your testing implementation here
                def test_rag_system():
                    # Create test queries
                    test_queries = [
                        "Your test question 1",
                        "Your test question 2",
                        # Add more test cases
                    ]
    
                    # Test your RAG system
                    for query in test_queries:
                        # Test and evaluate
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
                ## 🎉 Reflection and Next Steps
    
                **TODO:** After completing your implementation, reflect on:
    
                1. **What worked well?** What aspects of your RAG system are you most proud of?
    
                2. **What was challenging?** What parts were difficult to implement or debug?
    
                3. **How could you improve it?** What would you do differently or add next?
    
                4. **Performance observations:** How well did your system handle different types of queries?
    
                **Next Steps:**
                - Try the bonus challenges if you haven't already
                - Compare your approach with the solution notebook
                - Think about how you could scale this to larger document collections
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
