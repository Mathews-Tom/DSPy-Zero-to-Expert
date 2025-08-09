# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import (
        DSPyParameterPanel,
        DSPyResultViewer,
        get_config,
        setup_dspy_environment,
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        cleandoc,
        dspy,
        get_config,
        json,
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
            # 🔍 Module 03: Retrieval-Augmented Generation (RAG)

            **Duration:** 120-150 minutes  
            **Prerequisites:** Completed Modules 00-02

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Master RAG architecture and implementation patterns  
            - ✅ Build comprehensive document processing pipelines  
            - ✅ Integrate vector databases with DSPy retrievers  
            - ✅ Optimize retrieval and generation components  
            - ✅ Evaluate and monitor RAG system performance  

            ## 🧩 What is RAG?

            **Retrieval-Augmented Generation** combines:  
            - **Information Retrieval** - Finding relevant documents/passages  
            - **Language Generation** - Using retrieved context for responses  
            - **Dynamic Knowledge** - Access to external, up-to-date information  
            - **Factual Grounding** - Reducing hallucinations with real data  

            ## 🏗️ RAG Architecture

            Our comprehensive RAG system includes:  
            1. **Document Processing** - Text chunking, embedding generation  
            2. **Vector Storage** - Efficient similarity search and retrieval  
            3. **Retrieval Components** - DSPy retrievers with optimization  
            4. **Generation Pipeline** - Context-aware response generation  
            5. **Evaluation Framework** - Quality assessment and monitoring  

            Let's build a production-ready RAG system!
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
                ## ✅ RAG Implementation Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  
                - RAG components enabled!  

                Ready to build comprehensive RAG systems!
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
def _(available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 📄 Step 1: Document Processing Pipeline

                Let's start by building a comprehensive document processing system:
                """
            )
        )

        # Document Processing Classes
        class DocumentProcessor:
            """Comprehensive document processing for RAG systems."""

            def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.processed_docs = []

            def process_text(self, text: str, metadata: dict = None) -> list[dict]:
                """Process text into chunks with metadata."""
                chunks = self._chunk_text(text)
                processed_chunks = []

                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "id": f"chunk_{len(self.processed_docs)}_{i}",
                        "text": chunk,
                        "metadata": metadata or {},
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "processed_at": time.time(),
                    }
                    processed_chunks.append(chunk_data)
                    self.processed_docs.append(chunk_data)

                return processed_chunks

            def process_documents(self, documents: list[dict]) -> list[dict]:
                """Process multiple documents."""
                all_chunks = []
                for doc in documents:
                    text = doc.get("content", "")
                    metadata = {
                        "title": doc.get("title", "Unknown"),
                        "source": doc.get("source", "Unknown"),
                        "doc_type": doc.get("type", "text"),
                        **doc.get("metadata", {}),
                    }
                    chunks = self.process_text(text, metadata)
                    all_chunks.extend(chunks)

                return all_chunks

            def _chunk_text(self, text: str) -> list[str]:
                """Split text into overlapping chunks."""
                if len(text) <= self.chunk_size:
                    return [text]

                chunks = []
                start = 0

                while start < len(text):
                    end = start + self.chunk_size

                    # Try to break at sentence boundary
                    if end < len(text):
                        # Look for sentence endings
                        for i in range(
                            end, max(start + self.chunk_size // 2, end - 100), -1
                        ):
                            if text[i] in ".!?":
                                end = i + 1
                                break

                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)

                    start = end - self.chunk_overlap
                    if start >= len(text):
                        break

                return chunks

            def get_statistics(self) -> dict:
                """Get processing statistics."""
                if not self.processed_docs:
                    return {"total_chunks": 0, "avg_chunk_length": 0}

                total_chunks = len(self.processed_docs)
                avg_length = (
                    sum(len(doc["text"]) for doc in self.processed_docs) / total_chunks
                )

                return {
                    "total_chunks": total_chunks,
                    "avg_chunk_length": avg_length,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }

        # Create document processor
        doc_processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)

        cell3_content = mo.md(
            cleandoc(
                """
                ### 📄 Document Processor Created

                **Processing Features:**  
                - **Smart Chunking** - Sentence-boundary aware text splitting  
                - **Metadata Preservation** - Rich metadata tracking  
                - **Overlap Management** - Configurable chunk overlap  
                - **Statistics Tracking** - Processing metrics and analytics  

                The processor is ready to handle various document types!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        DocumentProcessor = None
        doc_processor = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (doc_processor,)


@app.cell
def _(available_providers, cleandoc, doc_processor, mo, output):
    if available_providers and doc_processor:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🗄️ Step 2: Vector Database Integration

                Let's build a vector database system for efficient similarity search:
                """
            )
        )

        # Simple Vector Database Implementation
        class SimpleVectorDB:
            """Simple in-memory vector database for RAG demonstrations."""

            def __init__(self):
                self.documents = []
                self.embeddings = []
                self.metadata = []
                self.index_map = {}

            def add_documents(
                self, docs: list[dict], embeddings: list[list[float]] = None
            ):
                """Add documents with optional embeddings."""
                start_idx = len(self.documents)

                for i, doc in enumerate(docs):
                    doc_id = doc.get("id", f"doc_{start_idx + i}")
                    self.documents.append(doc["text"])
                    self.metadata.append(doc.get("metadata", {}))
                    self.index_map[doc_id] = start_idx + i

                    # Generate simple embeddings if not provided
                    if embeddings and i < len(embeddings):
                        self.embeddings.append(embeddings[i])
                    else:
                        # Simple word-based embedding (for demo purposes)
                        embedding = self._generate_simple_embedding(doc["text"])
                        self.embeddings.append(embedding)

            def search(self, query: str, top_k: int = 5) -> list[dict]:
                """Search for similar documents."""
                if not self.documents:
                    return []

                query_embedding = self._generate_simple_embedding(query)
                similarities = []

                for i, doc_embedding in enumerate(self.embeddings):
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    similarities.append((similarity, i))

                # Sort by similarity and get top_k
                similarities.sort(reverse=True)
                results = []

                for similarity, idx in similarities[:top_k]:
                    results.append(
                        {
                            "text": self.documents[idx],
                            "metadata": self.metadata[idx],
                            "similarity": similarity,
                            "index": idx,
                        }
                    )

                return results

            def _generate_simple_embedding(self, text: str) -> list[float]:
                """Generate simple word-based embedding (demo purposes only)."""
                # Simple bag-of-words approach with common words
                common_words = [
                    "the",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "from",
                    "up",
                    "about",
                    "into",
                    "through",
                    "during",
                    "before",
                    "after",
                    "above",
                    "below",
                    "between",
                    "among",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "being",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "must",
                    "can",
                    "this",
                    "that",
                    "these",
                    "those",
                    "a",
                    "an",
                ]

                words = text.lower().split()
                embedding = [0.0] * 50  # 50-dimensional embedding

                for i, word in enumerate(common_words[:50]):
                    embedding[i] = words.count(word) / len(words) if words else 0

                # Add some content-based features
                embedding[45] = len(words) / 100  # Document length feature
                embedding[46] = (
                    len(set(words)) / len(words) if words else 0
                )  # Vocabulary diversity
                embedding[47] = (
                    sum(1 for w in words if w.isupper()) / len(words) if words else 0
                )  # Uppercase ratio
                embedding[48] = (
                    sum(1 for w in words if any(c.isdigit() for c in w)) / len(words)
                    if words
                    else 0
                )  # Number ratio
                embedding[49] = (
                    sum(1 for w in words if len(w) > 6) / len(words) if words else 0
                )  # Long word ratio

                return embedding

            def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
                """Calculate cosine similarity between two vectors."""
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5

                if magnitude1 == 0 or magnitude2 == 0:
                    return 0

                return dot_product / (magnitude1 * magnitude2)

            def get_stats(self) -> dict:
                """Get database statistics."""
                return {
                    "total_documents": len(self.documents),
                    "embedding_dimension": (
                        len(self.embeddings[0]) if self.embeddings else 0
                    ),
                    "avg_doc_length": (
                        sum(len(doc) for doc in self.documents) / len(self.documents)
                        if self.documents
                        else 0
                    ),
                }

        # Create vector database
        vector_db = SimpleVectorDB()

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🗄️ Vector Database Created

                **Database Features:**  
                - **Document Storage** - Efficient text and metadata storage  
                - **Similarity Search** - Cosine similarity-based retrieval  
                - **Simple Embeddings** - Word-based embeddings for demonstration  
                - **Statistics Tracking** - Database metrics and analytics  

                *Note: In production, use proper embedding models like OpenAI embeddings or sentence-transformers*

                The vector database is ready for document indexing!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        SimpleVectorDB = None
        vector_db = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (vector_db,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output, time):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🔍 Step 3: DSPy RAG Components

                Now let's build DSPy-integrated RAG components:
                """
            )
        )

        # DSPy RAG Signatures
        class RAGRetrievalSignature(dspy.Signature):
            """Retrieve relevant documents for a given query."""

            query = dspy.InputField(desc="User query or question")
            context_type = dspy.InputField(
                desc="Type of context needed: factual, analytical, creative"
            )
            retrieved_docs = dspy.OutputField(
                desc="Most relevant documents for the query"
            )
            retrieval_reasoning = dspy.OutputField(
                desc="Reasoning for document selection"
            )
            confidence = dspy.OutputField(
                desc="Confidence in retrieval quality (0.0-1.0)"
            )

        class RAGGenerationSignature(dspy.Signature):
            """Generate response using retrieved context."""

            query = dspy.InputField(desc="Original user query")
            retrieved_context = dspy.InputField(desc="Retrieved documents and passages")
            context_relevance = dspy.InputField(desc="Assessment of context relevance")
            response = dspy.OutputField(
                desc="Comprehensive response using retrieved context"
            )
            source_citations = dspy.OutputField(
                desc="Citations and references to sources used"
            )
            confidence = dspy.OutputField(
                desc="Confidence in response accuracy (0.0-1.0)"
            )

        class RAGEvaluationSignature(dspy.Signature):
            """Evaluate RAG system performance."""

            query = dspy.InputField(desc="Original query")
            retrieved_context = dspy.InputField(desc="Retrieved context")
            generated_response = dspy.InputField(desc="Generated response")
            relevance_score = dspy.OutputField(desc="Context relevance score (0.0-1.0)")
            accuracy_score = dspy.OutputField(desc="Response accuracy score (0.0-1.0)")
            completeness_score = dspy.OutputField(
                desc="Response completeness score (0.0-1.0)"
            )
            improvement_suggestions = dspy.OutputField(
                desc="Specific suggestions for improvement"
            )

        # RAG Pipeline Class
        class DSPyRAGPipeline:
            """Complete DSPy-based RAG pipeline."""

            def __init__(self, vector_db, doc_processor):
                self.vector_db = vector_db
                self.doc_processor = doc_processor
                self.retriever = dspy.ChainOfThought(RAGRetrievalSignature)
                self.generator = dspy.ChainOfThought(RAGGenerationSignature)
                self.evaluator = dspy.ChainOfThought(RAGEvaluationSignature)
                self.query_history = []

            def add_documents(self, documents: list[dict]):
                """Add documents to the RAG system."""
                # Process documents
                processed_chunks = self.doc_processor.process_documents(documents)

                # Add to vector database
                self.vector_db.add_documents(processed_chunks)

                return {
                    "processed_chunks": len(processed_chunks),
                    "total_documents": len(self.vector_db.documents),
                }

            def query(
                self, query: str, context_type: str = "factual", top_k: int = 3
            ) -> dict:
                """Execute RAG query pipeline."""
                start_time = time.time()

                try:
                    # Step 1: Retrieve relevant documents
                    retrieved_docs = self.vector_db.search(query, top_k=top_k)

                    # Format retrieved context
                    context_text = "\n\n".join(
                        [
                            f"Document {i+1} (similarity: {doc['similarity']:.3f}):\n{doc['text']}"
                            for i, doc in enumerate(retrieved_docs)
                        ]
                    )

                    # Step 2: Use DSPy retriever for reasoning
                    retrieval_result = self.retriever(
                        query=query, context_type=context_type
                    )

                    # Step 3: Generate response using retrieved context
                    generation_result = self.generator(
                        query=query,
                        retrieved_context=context_text,
                        context_relevance=retrieval_result.confidence,
                    )

                    # Step 4: Evaluate the result
                    evaluation_result = self.evaluator(
                        query=query,
                        retrieved_context=context_text,
                        generated_response=generation_result.response,
                    )

                    execution_time = time.time() - start_time

                    # Store query history
                    query_record = {
                        "query": query,
                        "context_type": context_type,
                        "retrieved_docs": len(retrieved_docs),
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                    }
                    self.query_history.append(query_record)

                    return {
                        "success": True,
                        "query": query,
                        "retrieved_documents": retrieved_docs,
                        "retrieval_reasoning": retrieval_result.retrieval_reasoning,
                        "retrieval_confidence": retrieval_result.confidence,
                        "response": generation_result.response,
                        "source_citations": generation_result.source_citations,
                        "generation_confidence": generation_result.confidence,
                        "evaluation": {
                            "relevance_score": evaluation_result.relevance_score,
                            "accuracy_score": evaluation_result.accuracy_score,
                            "completeness_score": evaluation_result.completeness_score,
                            "improvement_suggestions": evaluation_result.improvement_suggestions,
                        },
                        "execution_time": execution_time,
                        "metadata": {
                            "top_k": top_k,
                            "context_type": context_type,
                            "total_docs_in_db": len(self.vector_db.documents),
                        },
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "query": query,
                        "execution_time": time.time() - start_time,
                    }

            def get_analytics(self) -> dict:
                """Get RAG system analytics."""
                if not self.query_history:
                    return {"total_queries": 0}

                total_queries = len(self.query_history)
                avg_execution_time = (
                    sum(q["execution_time"] for q in self.query_history) / total_queries
                )

                return {
                    "total_queries": total_queries,
                    "avg_execution_time": avg_execution_time,
                    "database_stats": self.vector_db.get_stats(),
                    "processor_stats": self.doc_processor.get_statistics(),
                }

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🔍 DSPy RAG Components Created

                **RAG Pipeline Features:**  
                - **Intelligent Retrieval** - DSPy-powered document selection with reasoning  
                - **Context-Aware Generation** - Response generation using retrieved context  
                - **Automatic Evaluation** - Built-in quality assessment and improvement suggestions  
                - **Analytics Tracking** - Performance monitoring and query analytics  

                The complete RAG pipeline is ready for testing!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        RAGRetrievalSignature = None
        RAGGenerationSignature = None
        RAGEvaluationSignature = None
        DSPyRAGPipeline = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (DSPyRAGPipeline,)


@app.cell
def _(
    DSPyRAGPipeline,
    available_providers,
    cleandoc,
    doc_processor,
    mo,
    output,
    vector_db,
):
    if available_providers and DSPyRAGPipeline:
        # Create RAG pipeline
        rag_pipeline = DSPyRAGPipeline(vector_db, doc_processor)

        # Sample documents for testing
        sample_documents = [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through interaction with an environment.",
                "source": "ML Textbook",
                "type": "educational",
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain. Deep learning has revolutionized fields such as computer vision, natural language processing, and speech recognition. Key architectures include convolutional neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for sequential data, and transformers for language understanding.",
                "source": "Deep Learning Guide",
                "type": "educational",
            },
            {
                "title": "Natural Language Processing Applications",
                "content": "Natural Language Processing (NLP) is a field that combines computational linguistics with machine learning to help computers understand, interpret, and generate human language. Modern NLP applications include chatbots, language translation, sentiment analysis, text summarization, and question-answering systems. Recent advances in transformer models like BERT, GPT, and T5 have significantly improved the performance of NLP tasks. These models can understand context, generate coherent text, and perform complex language reasoning tasks.",
                "source": "NLP Research Paper",
                "type": "research",
            },
            {
                "title": "AI Ethics and Responsible Development",
                "content": "As artificial intelligence becomes more prevalent in society, it's crucial to consider the ethical implications of AI systems. Key concerns include bias in AI models, privacy protection, transparency in decision-making, and the potential impact on employment. Responsible AI development involves ensuring fairness, accountability, and transparency in AI systems. Organizations must implement ethical guidelines, conduct bias audits, and ensure that AI systems are developed and deployed in ways that benefit society while minimizing harm.",
                "source": "AI Ethics Journal",
                "type": "policy",
            },
            {
                "title": "Computer Vision Techniques",
                "content": "Computer vision is a field of AI that enables computers to interpret and understand visual information from the world. It involves techniques for image processing, object detection, facial recognition, and scene understanding. Modern computer vision systems use deep learning models, particularly convolutional neural networks, to achieve human-level performance in many visual tasks. Applications include autonomous vehicles, medical image analysis, security systems, and augmented reality. Key challenges include handling variations in lighting, perspective, and occlusion.",
                "source": "Computer Vision Handbook",
                "type": "technical",
            },
        ]

        # Add documents to RAG system
        add_result = rag_pipeline.add_documents(sample_documents)

        cell6_out = mo.vstack(
            [
                mo.md("### 📚 Sample Knowledge Base Created"),
                mo.md(f"**Documents Added:** {len(sample_documents)}"),
                mo.md(f"**Processed Chunks:** {add_result['processed_chunks']}"),
                mo.md(f"**Total Documents in DB:** {add_result['total_documents']}"),
                mo.md(
                    cleandoc(
                        """
                        **Knowledge Base Topics:**  
                        - Machine Learning Fundamentals  
                        - Deep Learning and Neural Networks  
                        - Natural Language Processing  
                        - AI Ethics and Responsible Development  
                        - Computer Vision Techniques  

                        The RAG system is ready for queries!
                        """
                    )
                ),
            ]
        )
    else:
        rag_pipeline = None
        sample_documents = None
        add_result = None
        cell6_out = mo.md("*RAG pipeline not available*")

    output.replace(cell6_out)
    return (rag_pipeline,)


@app.cell
def _(available_providers, mo, output, rag_pipeline):
    if available_providers and rag_pipeline:
        # Test queries for RAG system
        test_queries = [
            "What is machine learning and how does it work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the main applications of natural language processing?",
            "What ethical concerns should we consider when developing AI systems?",
            "How do convolutional neural networks work in computer vision?",
        ]

        query_selector = mo.ui.dropdown(
            options=test_queries,
            label="Select a query to test the RAG system",
            value=test_queries[0],
        )

        context_type_selector = mo.ui.dropdown(
            options=["factual", "analytical", "creative"],
            label="Context Type",
            value="factual",
        )

        top_k_slider = mo.ui.slider(
            start=1, stop=5, value=3, label="Number of documents to retrieve (top_k)"
        )

        run_rag_query = mo.ui.run_button(label="🔍 Run RAG Query")

        cell7_out = mo.vstack(
            [
                mo.md("### 🧪 RAG System Testing"),
                mo.md("Test the complete RAG pipeline with different queries:  "),
                query_selector,
                mo.hstack([context_type_selector, top_k_slider]),
                run_rag_query,
            ]
        )
    else:
        test_queries = None
        query_selector = None
        context_type_selector = None
        top_k_slider = None
        run_rag_query = None
        cell7_out = mo.md("*RAG pipeline not available*")

    output.replace(cell7_out)
    return context_type_selector, query_selector, run_rag_query, top_k_slider


@app.cell
def _(
    available_providers,
    cleandoc,
    context_type_selector,
    json,
    mo,
    output,
    query_selector,
    rag_pipeline,
    run_rag_query,
    top_k_slider,
):
    if available_providers and run_rag_query.value and rag_pipeline:
        try:
            selected_query = query_selector.value
            context_type = context_type_selector.value
            top_k = top_k_slider.value

            # Execute RAG query
            rag_result = rag_pipeline.query(
                query=selected_query, context_type=context_type, top_k=top_k
            )

            if rag_result["success"]:
                # Display comprehensive results
                retrieved_docs = rag_result["retrieved_documents"]

                # Format retrieved documents
                doc_displays = []
                for i, doc in enumerate(retrieved_docs):
                    doc_displays.append(
                        cleandoc(
                            f"""
                            **Document {i+1}** (Similarity: {doc['similarity']:.3f})  
                            - **Source:** {doc['metadata'].get('source', 'Unknown')}  
                            - **Type:** {doc['metadata'].get('doc_type', 'Unknown')}  
                            - **Content:** {doc['text'][:200]}{'...' if len(doc['text']) > 200 else ''}  
                            """
                        )
                    )

                cell8_out = mo.vstack(
                    [
                        mo.md("## 🔍 RAG Query Results"),
                        mo.md(f"**Query:** {selected_query}"),
                        mo.md(f"**Context Type:** {context_type}"),
                        mo.md(f"**Top-K:** {top_k}"),
                        mo.md(
                            f"**Execution Time:** {rag_result['execution_time']:.3f}s"
                        ),
                        mo.md("### 📄 Retrieved Documents"),
                        mo.md("\n".join(doc_displays)),
                        mo.md("### 🧠 Retrieval Analysis"),
                        mo.md(f"**Reasoning:** {rag_result['retrieval_reasoning']}"),
                        mo.md(f"**Confidence:** {rag_result['retrieval_confidence']}"),
                        mo.md("### 🎯 Generated Response"),
                        mo.md(f"**Response:** {rag_result['response']}"),
                        mo.md(
                            f"**Source Citations:** {rag_result['source_citations']}"
                        ),
                        mo.md(
                            f"**Generation Confidence:** {rag_result['generation_confidence']}"
                        ),
                        mo.md("### 📊 Quality Evaluation"),
                        mo.md(
                            f"**Relevance Score:** {rag_result['evaluation']['relevance_score']}"
                        ),
                        mo.md(
                            f"**Accuracy Score:** {rag_result['evaluation']['accuracy_score']}"
                        ),
                        mo.md(
                            f"**Completeness Score:** {rag_result['evaluation']['completeness_score']}"
                        ),
                        mo.md(
                            f"**Improvement Suggestions:** {rag_result['evaluation']['improvement_suggestions']}"
                        ),
                        mo.md("### 🔧 System Metadata"),
                        mo.md(
                            f"```json\n{json.dumps(rag_result['metadata'], indent=2)}\n```"
                        ),
                    ]
                )
            else:
                cell8_out = mo.md(
                    cleandoc(
                        f"""
                        ## ❌ RAG Query Failed

                        **Error:** {rag_result['error']}  
                        **Query:** {selected_query}  
                        **Execution Time:** {rag_result['execution_time']:.3f}s  
                        """
                    )
                )

        except Exception as e:
            cell8_out = mo.md(f"❌ **RAG Query Error:** {str(e)}")
    else:
        cell8_out = mo.md(
            "*Select a query and click 'Run RAG Query' to test the system*"
        )

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output, rag_pipeline):
    if available_providers and rag_pipeline:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 4: RAG Analytics Dashboard

                Let's analyze the performance of our RAG system:
                """
            )
        )

        # Analytics button
        show_analytics = mo.ui.run_button(label="📊 Show RAG Analytics")

        cell9_content = mo.vstack(
            [
                mo.md("### 📈 RAG System Analytics"),
                mo.md("View comprehensive analytics for the RAG system:"),
                show_analytics,
            ]
        )
    else:
        cell9_desc = mo.md("*RAG pipeline not available*")
        show_analytics = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack([cell9_desc, cell9_content])
    output.replace(cell9_out)
    return (show_analytics,)


@app.cell
def _(available_providers, cleandoc, mo, output, rag_pipeline, show_analytics):
    if available_providers and show_analytics.value and rag_pipeline:
        try:
            # Get analytics
            analytics = rag_pipeline.get_analytics()

            if analytics["total_queries"] > 0:
                # Create analytics display
                cell10_out = mo.vstack(
                    [
                        mo.md("## 📊 RAG System Analytics"),
                        mo.md("### 🎯 Query Performance"),
                        mo.md(
                            cleandoc(
                                f"""
                                - **Total Queries:** {analytics['total_queries']}  
                                - **Average Execution Time:** {analytics['avg_execution_time']:.3f}s  
                                - **Queries per Second:** {1/analytics['avg_execution_time']:.2f}  
                                """
                            )
                        ),
                        mo.md("### 🗄️ Database Statistics"),
                        mo.md(
                            cleandoc(
                                f"""
                                - **Total Documents:** {analytics['database_stats']['total_documents']}  
                                - **Embedding Dimension:** {analytics['database_stats']['embedding_dimension']}  
                                - **Average Document Length:** {analytics['database_stats']['avg_doc_length']:.1f} characters  
                                """
                            )
                        ),
                        mo.md("### 📄 Processing Statistics"),
                        mo.md(
                            cleandoc(
                                f"""
                                - **Total Chunks:** {analytics['processor_stats']['total_chunks']}  
                                - **Average Chunk Length:** {analytics['processor_stats']['avg_chunk_length']:.1f} characters  
                                - **Chunk Size:** {analytics['processor_stats']['chunk_size']}  
                                - **Chunk Overlap:** {analytics['processor_stats']['chunk_overlap']}  
                                """
                            )
                        ),
                        mo.md(
                            cleandoc(
                                """
                                ### 💡 Performance Insights

                                **Optimization Opportunities:**  
                                - Monitor query execution times for performance bottlenecks  
                                - Adjust chunk size based on query complexity  
                                - Consider caching for frequently asked questions  
                                - Implement query preprocessing for better retrieval  

                                **Quality Improvements:**  
                                - Use proper embedding models (OpenAI, sentence-transformers)  
                                - Implement semantic chunking strategies  
                                - Add query expansion and reformulation  
                                - Implement relevance feedback mechanisms  
                                """
                            )
                        ),
                    ]
                )
            else:
                cell10_out = mo.md(
                    "📊 **No queries executed yet.** Run some RAG queries first to see analytics."
                )

        except Exception as e:
            cell10_out = mo.md(f"❌ **Analytics Error:** {str(e)}")
    else:
        cell10_out = mo.md(
            "*Click 'Show RAG Analytics' to view system performance metrics*"
        )

    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell11_out = mo.md(
        cleandoc(
            """
            ## 🎓 RAG Implementation Module Complete!

            ### 🏆 What You've Mastered

            ✅ **RAG Architecture** - Complete understanding of retrieval-augmented generation  
            ✅ **Document Processing** - Smart chunking and metadata management  
            ✅ **Vector Database Integration** - Similarity search and document storage  
            ✅ **DSPy RAG Components** - Intelligent retrieval and generation with reasoning  
            ✅ **Quality Evaluation** - Automatic assessment and improvement suggestions  
            ✅ **Performance Analytics** - Comprehensive monitoring and optimization insights  

            ### 🛠️ Key Components Built

            1. **DocumentProcessor**  
                - Smart text chunking with sentence boundary awareness  
                - Rich metadata preservation and tracking  
                - Configurable chunk size and overlap  
                - Processing statistics and analytics  

            2. **SimpleVectorDB**  
                - In-memory vector storage and similarity search  
                - Cosine similarity-based retrieval  
                - Document metadata management  
                - Performance statistics tracking  

            3. **DSPyRAGPipeline**  
                - Complete RAG workflow with DSPy integration  
                - Intelligent retrieval with reasoning  
                - Context-aware response generation  
                - Automatic quality evaluation and improvement suggestions  

            4. **Analytics Dashboard**  
                - Query performance monitoring  
                - Database and processing statistics  
                - Optimization recommendations  
                - Quality insights and improvements  

            ### 🎯 Skills Developed

            - **RAG System Design** - Architecture patterns for retrieval-augmented generation
            - **Document Processing** - Text chunking and preprocessing strategies
            - **Vector Search** - Similarity-based information retrieval
            - **DSPy Integration** - Combining retrieval with language model reasoning
            - **Quality Assessment** - Evaluating and improving RAG system performance

            ### 🚀 Ready for Advanced RAG Topics?

            You now understand comprehensive RAG implementation! Time to explore advanced techniques:

            **Next Module:**
            ```bash
            uv run marimo run 03-retrieval-rag/vector_database_setup.py
            ```

            **Coming Up:**
            - Multiple vector database integrations (FAISS, ChromaDB, Qdrant)
            - Advanced embedding strategies and models
            - Retrieval optimization and ranking algorithms
            - Production deployment and scaling considerations

            ### 💡 Practice Challenges

            Before moving on, try extending the RAG system:

            1. **Advanced Document Types**
                - PDF processing with layout awareness
                - Structured data integration (tables, lists)
                - Multi-modal content (text + images)

            2. **Retrieval Improvements**
                - Hybrid search (keyword + semantic)
                - Query expansion and reformulation
                - Re-ranking and relevance scoring

            3. **Generation Enhancements**
                - Multi-document synthesis
                - Citation and source tracking
                - Fact-checking and verification

            4. **Production Features**
                - Caching and performance optimization
                - Real-time document updates
                - User feedback integration

            ### 🏭 Production Considerations

            When deploying RAG systems:  
            - **Embedding Models**: Use production-grade embeddings (OpenAI, sentence-transformers)  
            - **Vector Databases**: Implement proper vector databases (Pinecone, Weaviate, Qdrant)  
            - **Scalability**: Design for large document collections and high query volumes  
            - **Quality Monitoring**: Implement continuous evaluation and improvement  
            - **Security**: Ensure proper access control and data privacy  

            Master these RAG patterns and you can build sophisticated knowledge-augmented AI systems!
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell11_out)
    return


if __name__ == "__main__":
    app.run()
