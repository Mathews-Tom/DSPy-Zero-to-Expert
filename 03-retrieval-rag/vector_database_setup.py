# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():

    import json
    import random
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from common import get_config, setup_dspy_environment

    return (
        cleandoc,
        get_config,
        json,
        mo,
        output,
        random,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🗄️ Vector Database Setup and Management

            **Duration:** 90-120 minutes  
            **Prerequisites:** Completed RAG Implementation

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Master multiple vector database integrations  
            - ✅ Implement FAISS, ChromaDB, and Qdrant connections  
            - ✅ Build database initialization and data loading tools  
            - ✅ Create performance monitoring and optimization systems  
            - ✅ Design scalable vector database architectures  

            ## 🗄️ Vector Database Landscape

            **Vector Databases** provide:  
            - **Efficient Storage** - Optimized for high-dimensional vectors  
            - **Fast Similarity Search** - Sub-linear search performance  
            - **Scalability** - Handle millions to billions of vectors  
            - **Persistence** - Durable storage with backup and recovery  
            - **Metadata Filtering** - Combine vector search with traditional filters  

            ## 🏗️ Database Options

            We'll implement integrations for:  
            1. **FAISS** - Facebook's similarity search library  
            2. **ChromaDB** - Open-source embedding database  
            3. **Qdrant** - Vector similarity search engine  
            4. **Unified Interface** - Common API across all databases  
            5. **Performance Comparison** - Benchmarking and optimization  

            Let's build a comprehensive vector database management system!
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
                ## ✅ Vector Database Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  
                - Vector database integrations enabled!  

                Ready to build comprehensive database management!
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
                ## 🏗️ Step 1: Unified Vector Database Interface

                Let's create a unified interface for multiple vector databases:
                """
            )
        )

        # Abstract Vector Database Interface
        from abc import ABC, abstractmethod

        class VectorDatabaseInterface(ABC):
            """Abstract interface for vector databases."""

            @abstractmethod
            def initialize(self, config: dict) -> bool:
                """Initialize the database connection."""

            @abstractmethod
            def add_vectors(
                self,
                vectors: list[list[float]],
                metadata: list[dict],
                ids: list[str] = None,
            ) -> bool:
                """Add vectors with metadata to the database."""

            @abstractmethod
            def search(
                self, _query_vector: list[float], _top_k: int = 10, filters: dict = None
            ) -> list[dict]:
                """Search for similar vectors."""

            @abstractmethod
            def delete_vectors(self, ids: list[str]) -> bool:
                """Delete vectors by IDs."""

            @abstractmethod
            def get_stats(self) -> dict:
                """Get database statistics."""

            @abstractmethod
            def close(self) -> bool:
                """Close database connection."""

        # Database Manager
        class VectorDatabaseManager:
            """Manages multiple vector database implementations."""

            def __init__(self):
                self.databases = {}
                self.active_db = None
                self.performance_metrics = {}

            def register_database(
                self, name: str, db_instance: VectorDatabaseInterface
            ):
                """Register a vector database implementation."""
                self.databases[name] = db_instance
                self.performance_metrics[name] = {
                    "total_operations": 0,
                    "total_time": 0.0,
                    "avg_latency": 0.0,
                    "last_operation": None,
                }

            def set_active_database(self, name: str) -> bool:
                """Set the active database for operations."""
                if name in self.databases:
                    self.active_db = name
                    return True
                return False

            def execute_operation(self, operation: str, *args, **kwargs):
                """Execute operation on active database with performance tracking."""
                if not self.active_db or self.active_db not in self.databases:
                    raise ValueError("No active database set")

                db = self.databases[self.active_db]
                metrics = self.performance_metrics[self.active_db]

                _start_time = time.time()
                try:
                    _result = getattr(db, operation)(*args, **kwargs)
                    execution_time = time.time() - _start_time

                    # Update metrics
                    metrics["total_operations"] += 1
                    metrics["total_time"] += execution_time
                    metrics["avg_latency"] = (
                        metrics["total_time"] / metrics["total_operations"]
                    )
                    metrics["last_operation"] = {
                        "operation": operation,
                        "time": execution_time,
                        "timestamp": time.time(),
                    }

                    return _result
                except Exception as e:
                    execution_time = time.time() - _start_time
                    metrics["total_operations"] += 1
                    metrics["total_time"] += execution_time
                    raise e

            def get_performance_comparison(self) -> dict:
                """Compare performance across all databases."""
                comparison = {}
                for name, metrics in self.performance_metrics.items():
                    if metrics["total_operations"] > 0:
                        comparison[name] = {
                            "operations": metrics["total_operations"],
                            "avg_latency": metrics["avg_latency"],
                            "total_time": metrics["total_time"],
                        }
                return comparison

            def list_databases(self) -> list[str]:
                """list all registered databases."""
                return list(self.databases.keys())

        cell3_content = mo.md(
            """
        ### 🏗️ Unified Database Interface Created

        **Interface Features:**  
        - **Abstract Base Class** - Common interface for all vector databases  
        - **Database Manager** - Centralized management and switching  
        - **Performance Tracking** - Automatic latency and operation metrics  
        - **Operation Routing** - Seamless switching between database backends  

        The unified interface is ready for database implementations!
        """
        )
    else:
        cell3_desc = mo.md("")
        VectorDatabaseInterface = None
        VectorDatabaseManager = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return VectorDatabaseInterface, VectorDatabaseManager


@app.cell
def _(
    VectorDatabaseInterface,
    available_providers,
    cleandoc,
    mo,
    output,
    time,
):
    if available_providers and VectorDatabaseInterface:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 2: FAISS Database Implementation

                Let's implement FAISS (Facebook AI Similarity Search) integration:
                """
            )
        )

        # FAISS Implementation (Simulated)
        class FAISSDatabase(VectorDatabaseInterface):
            """FAISS vector database implementation."""

            def __init__(self):
                self.index = None
                self.metadata_store = {}
                self.id_to_index = {}
                self.index_to_id = {}
                self.dimension = None
                self.next_index = 0
                self.initialized = False

            def initialize(self, config: dict) -> bool:
                """Initialize FAISS index."""
                try:
                    self.dimension = config.get("dimension", 128)
                    index_type = config.get("index_type", "flat")

                    # Simulate FAISS index creation
                    # In real implementation: import faiss; self.index = faiss.IndexFlatIP(self.dimension)
                    self.index = {
                        "type": f"FAISS_{index_type}",
                        "dimension": self.dimension,
                        "vectors": [],
                        "created_at": time.time(),
                    }

                    self.initialized = True
                    return True
                except Exception as e:
                    print(f"FAISS initialization error: {e}")
                    return False

            def add_vectors(
                self,
                vectors: list[list[float]],
                metadata: list[dict],
                ids: list[str] = None,
            ) -> bool:
                """Add vectors to FAISS index."""
                if not self.initialized:
                    return False

                try:
                    for _i, vector in enumerate(vectors):
                        vector_id = (
                            ids[_i]
                            if ids and _i < len(ids)
                            else f"faiss_vec_{self.next_index}"
                        )

                        # Store vector (simulated)
                        self.index["vectors"].append(vector)

                        # Store metadata
                        self.metadata_store[vector_id] = (
                            metadata[_i] if _i < len(metadata) else {}
                        )

                        # Update mappings
                        self.id_to_index[vector_id] = self.next_index
                        self.index_to_id[self.next_index] = vector_id

                        self.next_index += 1

                    return True
                except Exception as e:
                    print(f"FAISS add_vectors error: {e}")
                    return False

            def search(
                self, _query_vector: list[float], _top_k: int = 10, filters: dict = None
            ) -> list[dict]:
                """Search FAISS index for similar vectors."""
                if not self.initialized or not self.index["vectors"]:
                    return []

                try:
                    # Simulate FAISS search with cosine similarity
                    similarities = []
                    for idx, stored_vector in enumerate(self.index["vectors"]):
                        similarity = self._cosine_similarity(
                            _query_vector, stored_vector
                        )
                        vector_id = self.index_to_id[idx]

                        # Apply filters if provided
                        if filters:
                            metadata = self.metadata_store.get(vector_id, {})
                            if not self._apply_filters(metadata, filters):
                                continue

                        similarities.append(
                            {
                                "id": vector_id,
                                "similarity": similarity,
                                "metadata": self.metadata_store.get(vector_id, {}),
                                "index": idx,
                            }
                        )

                    # Sort by similarity and return _top_k
                    similarities.sort(key=lambda x: x["similarity"], reverse=True)
                    return similarities[:_top_k]

                except Exception as e:
                    print(f"FAISS search error: {e}")
                    return []

            def delete_vectors(self, ids: list[str]) -> bool:
                """Delete vectors from FAISS index."""
                # Note: FAISS doesn't support deletion easily, this is simulated
                try:
                    for vector_id in ids:
                        if vector_id in self.id_to_index:
                            idx = self.id_to_index[vector_id]
                            # Mark as deleted (in real FAISS, you'd need to rebuild index)
                            self.metadata_store.pop(vector_id, None)
                            self.id_to_index.pop(vector_id, None)
                            self.index_to_id.pop(idx, None)
                    return True
                except Exception as e:
                    print(f"FAISS delete error: {e}")
                    return False

            def get_stats(self) -> dict:
                """Get FAISS database statistics."""
                return {
                    "database_type": "FAISS",
                    "total_vectors": len(self.index["vectors"]) if self.index else 0,
                    "dimension": self.dimension,
                    "index_type": self.index["type"] if self.index else None,
                    "initialized": self.initialized,
                    "memory_usage": (
                        len(self.index["vectors"]) * self.dimension * 4
                        if self.index
                        else 0
                    ),  # Approximate bytes
                }

            def close(self) -> bool:
                """Close FAISS database."""
                self.index = None
                self.metadata_store.clear()
                self.id_to_index.clear()
                self.index_to_id.clear()
                self.initialized = False
                return True

            def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
                """Calculate cosine similarity."""
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5

                if magnitude1 == 0 or magnitude2 == 0:
                    return 0

                return dot_product / (magnitude1 * magnitude2)

            def _apply_filters(self, metadata: dict, filters: dict) -> bool:
                """Apply metadata filters."""
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        return False
                return True

        cell4_content = mo.md(
            cleandoc(
                """
                ### 📊 FAISS Database Implementation Created

                **FAISS Features:**  
                - **High Performance** - Optimized for large-scale similarity search  
                - **Multiple Index Types** - Flat, IVF, HNSW, and more  
                - **Memory Efficient** - Compressed vector storage options  
                - **Batch Operations** - Efficient bulk vector operations  

                *Note: This is a simulated implementation. In production, install faiss-cpu or faiss-gpu*

                The FAISS database is ready for integration!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        FAISSDatabase = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (FAISSDatabase,)


@app.cell
def _(
    VectorDatabaseInterface,
    available_providers,
    cleandoc,
    mo,
    output,
    time,
):
    if available_providers and VectorDatabaseInterface:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🎨 Step 3: ChromaDB Implementation

                Let's implement ChromaDB integration:
                """
            )
        )

        # ChromaDB Implementation (Simulated)
        class ChromaDatabase(VectorDatabaseInterface):
            """ChromaDB vector database implementation."""

            def __init__(self):
                self.client = None
                self.collection = None
                self.collection_name = "default_collection"
                self.initialized = False
                self.documents = {}
                self.vectors = {}
                self.metadata = {}

            def initialize(self, config: dict) -> bool:
                """Initialize ChromaDB client and collection."""
                try:
                    self.collection_name = config.get(
                        "collection_name", "default_collection"
                    )

                    # Simulate ChromaDB client creation
                    # In real implementation: import chromadb; self.client = chromadb.Client()
                    self.client = {
                        "type": "ChromaDB_Client",
                        "created_at": time.time(),
                        "config": config,
                    }

                    # Simulate collection creation
                    # In real implementation: self.collection = self.client.create_collection(name=self.collection_name)
                    self.collection = {
                        "name": self.collection_name,
                        "created_at": time.time(),
                        "document_count": 0,
                    }

                    self.initialized = True
                    return True
                except Exception as e:
                    print(f"ChromaDB initialization error: {e}")
                    return False

            def add_vectors(
                self,
                vectors: list[list[float]],
                metadata: list[dict],
                ids: list[str] = None,
            ) -> bool:
                """Add vectors to ChromaDB collection."""
                if not self.initialized:
                    return False

                try:
                    for _i, vector in enumerate(vectors):
                        vector_id = (
                            ids[_i]
                            if ids and _i < len(ids)
                            else f"chroma_doc_{len(self.vectors)}"
                        )

                        # Store vector and metadata
                        self.vectors[vector_id] = vector
                        self.metadata[vector_id] = (
                            metadata[_i] if _i < len(metadata) else {}
                        )

                        # Create document text from metadata (ChromaDB stores documents)
                        doc_text = metadata[_i].get("text", f"Document {vector_id}")
                        self.documents[vector_id] = doc_text

                        self.collection["document_count"] += 1

                    return True
                except Exception as e:
                    print(f"ChromaDB add_vectors error: {e}")
                    return False

            def search(
                self, _query_vector: list[float], _top_k: int = 10, filters: dict = None
            ) -> list[dict]:
                """Search ChromaDB collection for similar vectors."""
                if not self.initialized or not self.vectors:
                    return []

                try:
                    # Simulate ChromaDB search
                    similarities = []
                    for vector_id, stored_vector in self.vectors.items():
                        similarity = self._cosine_similarity(
                            _query_vector, stored_vector
                        )
                        doc_metadata = self.metadata.get(vector_id, {})

                        # Apply filters if provided
                        if filters:
                            if not self._apply_filters(doc_metadata, filters):
                                continue

                        similarities.append(
                            {
                                "id": vector_id,
                                "similarity": similarity,
                                "metadata": doc_metadata,
                                "document": self.documents.get(vector_id, ""),
                                "distance": 1 - similarity,  # ChromaDB uses distance
                            }
                        )

                    # Sort by similarity and return _top_k
                    similarities.sort(key=lambda x: x["similarity"], reverse=True)
                    return similarities[:_top_k]

                except Exception as e:
                    print(f"ChromaDB search error: {e}")
                    return []

            def delete_vectors(self, ids: list[str]) -> bool:
                """Delete vectors from ChromaDB collection."""
                try:
                    for vector_id in ids:
                        self.vectors.pop(vector_id, None)
                        self.metadata.pop(vector_id, None)
                        self.documents.pop(vector_id, None)
                        if self.collection["document_count"] > 0:
                            self.collection["document_count"] -= 1
                    return True
                except Exception as e:
                    print(f"ChromaDB delete error: {e}")
                    return False

            def get_stats(self) -> dict:
                """Get ChromaDB statistics."""
                return {
                    "database_type": "ChromaDB",
                    "collection_name": self.collection_name,
                    "total_vectors": len(self.vectors),
                    "total_documents": len(self.documents),
                    "initialized": self.initialized,
                    "supports_metadata_filtering": True,
                    "supports_full_text_search": True,
                }

            def close(self) -> bool:
                """Close ChromaDB connection."""
                self.client = None
                self.collection = None
                self.vectors.clear()
                self.metadata.clear()
                self.documents.clear()
                self.initialized = False
                return True

            def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
                """Calculate cosine similarity."""
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5

                if magnitude1 == 0 or magnitude2 == 0:
                    return 0

                return dot_product / (magnitude1 * magnitude2)

            def _apply_filters(self, metadata: dict, filters: dict) -> bool:
                """Apply metadata filters."""
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        return False
                return True

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🎨 ChromaDB Implementation Created

                **ChromaDB Features:**  
                - **Document-Centric** - Stores documents with embeddings  
                - **Metadata Filtering** - Rich filtering capabilities  
                - **Full-Text Search** - Combines vector and text search  
                - **Easy Integration** - Simple Python API  

                *Note: This is a simulated implementation. In production, install chromadb*

                The ChromaDB database is ready for integration!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        ChromaDatabase = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (ChromaDatabase,)


@app.cell
def _(
    VectorDatabaseInterface,
    available_providers,
    cleandoc,
    mo,
    output,
    time,
):
    if available_providers and VectorDatabaseInterface:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## ⚡ Step 4: Qdrant Implementation

                Let's implement Qdrant vector search engine integration:
                """
            )
        )

        # Qdrant Implementation (Simulated)
        class QdrantDatabase(VectorDatabaseInterface):
            """Qdrant vector database implementation."""

            def __init__(self):
                self.client = None
                self.collection_name = "default_collection"
                self.vectors = {}
                self.payloads = {}
                self.initialized = False
                self.vector_config = None

            def initialize(self, config: dict) -> bool:
                """Initialize Qdrant client and collection."""
                try:
                    self.collection_name = config.get(
                        "collection_name", "default_collection"
                    )
                    vector_size = config.get("vector_size", 128)
                    distance_metric = config.get("distance", "cosine")

                    # Simulate Qdrant client creation
                    # In real implementation: from qdrant_client import QdrantClient
                    self.client = {
                        "type": "Qdrant_Client",
                        "host": config.get("host", "localhost"),
                        "port": config.get("port", 6333),
                        "created_at": time.time(),
                    }

                    # Simulate collection configuration
                    self.vector_config = {
                        "size": vector_size,
                        "distance": distance_metric,
                        "collection_name": self.collection_name,
                    }

                    self.initialized = True
                    return True
                except Exception as e:
                    print(f"Qdrant initialization error: {e}")
                    return False

            def add_vectors(
                self,
                vectors: list[list[float]],
                metadata: list[dict],
                ids: list[str] = None,
            ) -> bool:
                """Add vectors to Qdrant collection."""
                if not self.initialized:
                    return False

                try:
                    for _i, vector in enumerate(vectors):
                        # Generate ID if not provided
                        vector_id = (
                            ids[_i]
                            if ids and _i < len(ids)
                            else f"qdrant_point_{len(self.vectors)}"
                        )

                        # Store vector and payload (Qdrant terminology)
                        self.vectors[vector_id] = vector
                        self.payloads[vector_id] = (
                            metadata[_i] if _i < len(metadata) else {}
                        )

                    return True
                except Exception as e:
                    print(f"Qdrant add_vectors error: {e}")
                    return False

            def search(
                self, _query_vector: list[float], _top_k: int = 10, filters: dict = None
            ) -> list[dict]:
                """Search Qdrant collection for similar vectors."""
                if not self.initialized or not self.vectors:
                    return []

                try:
                    # Simulate Qdrant search
                    similarities = []
                    for vector_id, stored_vector in self.vectors.items():
                        # Calculate similarity based on distance metric
                        if self.vector_config["distance"] == "cosine":
                            similarity = self._cosine_similarity(
                                _query_vector, stored_vector
                            )
                            distance = 1 - similarity
                        elif self.vector_config["distance"] == "euclidean":
                            distance = self._euclidean_distance(
                                _query_vector, stored_vector
                            )
                            similarity = 1 / (
                                1 + distance
                            )  # Convert distance to similarity
                        else:
                            similarity = self._cosine_similarity(
                                _query_vector, stored_vector
                            )
                            distance = 1 - similarity

                        payload = self.payloads.get(vector_id, {})

                        # Apply filters if provided (Qdrant-style filtering)
                        if filters:
                            if not self._apply_qdrant_filters(payload, filters):
                                continue

                        similarities.append(
                            {
                                "id": vector_id,
                                "similarity": similarity,
                                "distance": distance,
                                "payload": payload,
                                "vector": stored_vector,
                            }
                        )

                    # Sort by similarity and return _top_k
                    similarities.sort(key=lambda x: x["similarity"], reverse=True)
                    return similarities[:_top_k]

                except Exception as e:
                    print(f"Qdrant search error: {e}")
                    return []

            def delete_vectors(self, ids: list[str]) -> bool:
                """Delete vectors from Qdrant collection."""
                try:
                    for vector_id in ids:
                        self.vectors.pop(vector_id, None)
                        self.payloads.pop(vector_id, None)
                    return True
                except Exception as e:
                    print(f"Qdrant delete error: {e}")
                    return False

            def get_stats(self) -> dict:
                """Get Qdrant database statistics."""
                return {
                    "database_type": "Qdrant",
                    "collection_name": self.collection_name,
                    "total_vectors": len(self.vectors),
                    "vector_size": (
                        self.vector_config["size"] if self.vector_config else 0
                    ),
                    "distance_metric": (
                        self.vector_config["distance"]
                        if self.vector_config
                        else "unknown"
                    ),
                    "initialized": self.initialized,
                    "supports_filtering": True,
                    "supports_payload": True,
                }

            def close(self) -> bool:
                """Close Qdrant connection."""
                self.client = None
                self.vectors.clear()
                self.payloads.clear()
                self.initialized = False
                return True

            def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
                """Calculate cosine similarity."""
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5

                if magnitude1 == 0 or magnitude2 == 0:
                    return 0

                return dot_product / (magnitude1 * magnitude2)

            def _euclidean_distance(
                self, vec1: list[float], vec2: list[float]
            ) -> float:
                """Calculate Euclidean distance."""
                return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

            def _apply_qdrant_filters(self, payload: dict, filters: dict) -> bool:
                """Apply Qdrant-style filters."""
                # Simplified filter implementation
                for key, value in filters.items():
                    if key not in payload:
                        return False

                    # Support different filter types
                    if isinstance(value, dict):
                        # Range filters, etc.
                        if "gte" in value and payload[key] < value["gte"]:
                            return False
                        if "lte" in value and payload[key] > value["lte"]:
                            return False
                    else:
                        # Exact match
                        if payload[key] != value:
                            return False

                return True

        cell6_content = mo.md(
            cleandoc(
                """
                ### ⚡ Qdrant Implementation Created

                **Qdrant Features:**  
                - **High Performance** - Rust-based vector search engine  
                - **Advanced Filtering** - Complex payload-based filtering  
                - **Multiple Distance Metrics** - Cosine, Euclidean, Dot product  
                - **Scalability** - Distributed and cloud-ready  

                *Note: This is a simulated implementation. In production, install qdrant-client*

                The Qdrant database is ready for integration!
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        QdrantDatabase = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (QdrantDatabase,)


@app.cell
def _(
    ChromaDatabase,
    FAISSDatabase,
    QdrantDatabase,
    VectorDatabaseManager,
    available_providers,
    cleandoc,
    mo,
    output,
):
    if available_providers and VectorDatabaseManager:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🔧 Step 5: Database Manager Setup

                Let's set up the database manager with all implementations:
                """
            )
        )

        # Create database manager and register all databases
        db_manager = VectorDatabaseManager()

        # Initialize database instances
        faiss_db = FAISSDatabase()
        chroma_db = ChromaDatabase()
        qdrant_db = QdrantDatabase()

        # Register databases
        db_manager.register_database("faiss", faiss_db)
        db_manager.register_database("chromadb", chroma_db)
        db_manager.register_database("qdrant", qdrant_db)

        # Database configuration
        db_configs = {
            "faiss": {"dimension": 128, "index_type": "flat"},
            "chromadb": {"collection_name": "test_collection"},
            "qdrant": {
                "collection_name": "test_collection",
                "vector_size": 128,
                "distance": "cosine",
            },
        }

        # Initialize all databases
        initialization_results = {}
        for _db_name in db_manager.list_databases():
            try:
                db_manager.set_active_database(_db_name)
                _result = db_manager.execute_operation(
                    "initialize", db_configs[_db_name]
                )
                initialization_results[_db_name] = (
                    "✅ Success" if _result else "❌ Failed"
                )
            except Exception as e:
                initialization_results[_db_name] = f"❌ Error: {str(e)}"

        cell7_content = mo.vstack(
            [
                mo.md("### 🔧 Database Manager Setup Complete"),
                mo.md("**Registered Databases:**"),
                mo.md(
                    "\n".join(
                        [
                            f"- **{db}**: {status}"
                            for db, status in initialization_results.items()
                        ]
                    )
                ),
                mo.md(
                    f"**Available Databases:** {', '.join(db_manager.list_databases())}"
                ),
                mo.md(
                    cleandoc(
                        """
                        **Manager Features:**
                        - **Multi-Database Support** - FAISS, ChromaDB, and Qdrant
                        - **Unified Interface** - Common API across all databases
                        - **Performance Tracking** - Automatic latency monitoring
                        - **Easy Switching** - Change databases without code changes
                        """
                    )
                ),
            ]
        )
    else:
        cell7_desc = mo.md("")
        db_manager = None
        faiss_db = None
        chroma_db = None
        qdrant_db = None
        db_configs = None
        initialization_results = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return (db_manager,)


@app.cell
def _(available_providers, cleandoc, db_manager, mo, output):
    if available_providers and db_manager:
        cell8_desc = mo.md(
            cleandoc(
                """
                ## 🧪 Step 6: Database Testing Interface

                Let's test and compare the different vector databases:
                """
            )
        )

        # Database testing interface - individual components to avoid _clone errors
        database_type = mo.ui.dropdown(
            options=["faiss", "chromadb", "qdrant"],
            label="Select Database to Test",
            value="faiss",
        )
        _operation = mo.ui.dropdown(
            options=[
                "Add Vectors",
                "Search Vectors",
                "Get Statistics",
                "Performance Test",
            ],
            label="Operation",
            value="Add Vectors",
        )
        _num_vectors = mo.ui.slider(
            start=5, stop=50, value=10, label="Number of Test Vectors"
        )
        vector_dimension = mo.ui.slider(
            start=64, stop=256, value=128, label="Vector Dimension"
        )
        _search_query = mo.ui.text(
            placeholder="Enter search query (for search operation)",
            label="Search Query",
        )
        _top_k = mo.ui.slider(start=1, stop=10, value=5, label="Top-K Results")
        run_test = mo.ui.run_button(label="🧪 Run Database Test")

        # Create test interface as a dictionary-like object with value property
        class TestInterface:
            def __init__(self, components):
                self.components = components

            @property
            def value(self):
                return {
                    "database_type": self.components["database_type"].value,
                    "operation": self.components["operation"].value,
                    "_num_vectors": self.components["_num_vectors"].value,
                    "vector_dimension": self.components["vector_dimension"].value,
                    "_search_query": self.components["_search_query"].value,
                    "_top_k": self.components["_top_k"].value,
                    "run_test": self.components["run_test"].value,
                }

        test_interface = TestInterface(
            {
                "database_type": database_type,
                "operation": _operation,
                "_num_vectors": _num_vectors,
                "vector_dimension": vector_dimension,
                "_search_query": _search_query,
                "_top_k": _top_k,
                "run_test": run_test,
            }
        )

        # Arrange components with mo.vstack
        test_interface_layout = mo.vstack(
            [
                database_type,
                _operation,
                _num_vectors,
                vector_dimension,
                _search_query,
                _top_k,
                run_test,
            ]
        )

        cell8_content = mo.vstack(
            [
                mo.md("### 🧪 Database Testing Interface"),
                mo.md("Test different operations across vector databases:"),
                test_interface_layout,
            ]
        )
    else:
        cell8_desc = mo.md("")
        database_type = None
        _operation = None
        _num_vectors = None
        vector_dimension = None
        _search_query = None
        _top_k = None
        run_test = None
        test_interface = None
        cell8_content = mo.md("")

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return (
        database_type,
        _operation,
        _num_vectors,
        vector_dimension,
        _search_query,
        _top_k,
        run_test,
        test_interface,
    )


@app.cell
def _(
    available_providers,
    cleandoc,
    db_manager,
    json,
    mo,
    output,
    random,
    test_interface,
    time,
):
    if (
        available_providers
        and test_interface.value
        and test_interface.value["run_test"]
    ):
        try:
            # Get test parameters
            db_type = test_interface.value["database_type"]
            operation = test_interface.value["operation"]
            _num_vectors = test_interface.value["_num_vectors"]
            vector_dim = test_interface.value["vector_dimension"]
            _search_query = test_interface.value["_search_query"]
            _top_k = test_interface.value["_top_k"]

            # Set active database
            db_manager.set_active_database(db_type)

            if operation == "Add Vectors":
                _test_vectors = []
                _test_metadata = []
                _test_ids = []

                for _i in range(_num_vectors):
                    # Generate random vector
                    vector = [random.random() for _ in range(vector_dim)]
                    _test_vectors.append(vector)

                    # Generate metadata
                    metadata = {
                        "text": f"Test document {_i}",
                        "category": random.choice(["tech", "science", "business"]),
                        "score": random.uniform(0.1, 1.0),
                        "created_at": time.time(),
                    }
                    _test_metadata.append(metadata)
                    _test_ids.append(f"test_doc_{_i}")

                # Add vectors to database
                _start_time = time.time()
                _result = db_manager.execute_operation(
                    "add_vectors", _test_vectors, _test_metadata, _test_ids
                )
                execution_time = time.time() - _start_time

                cell9_out = mo.vstack(
                    [
                        mo.md(f"## 📊 Add Vectors Test Results ({db_type.upper()})"),
                        mo.md(f"**Operation:** {operation}"),
                        mo.md(f"**Vectors Added:** {_num_vectors}"),
                        mo.md(f"**Vector Dimension:** {vector_dim}"),
                        mo.md(f"**Execution Time:** {execution_time:.3f}s"),
                        mo.md(f"**Success:** {'✅ Yes' if _result else '❌ No'}"),
                        mo.md(
                            f"**Throughput:** {_num_vectors/execution_time:.1f} vectors/second"
                        ),
                    ]
                )

            elif operation == "Search Vectors":
                if not _search_query:
                    mo.md("❌ Please provide a search query")
                else:
                    # Generate query vector (simplified - in practice, use embedding model)
                    _query_vector = [
                        hash(_search_query + str(_i)) % 100 / 100.0
                        for _i in range(vector_dim)
                    ]

                    # Search database
                    _start_time = time.time()
                    _results = db_manager.execute_operation(
                        "search", _query_vector, _top_k
                    )
                    execution_time = time.time() - _start_time

                    # Format _results
                    result_displays = []
                    for _i, _result in enumerate(_results[:5]):  # Show top 5
                        result_displays.append(
                            cleandoc(
                                f"""
                                **Result {_i+1}:**
                                - **ID:** {_result.get('id', 'Unknown')}
                                - **Similarity:** {_result.get('similarity', 0):.3f}
                                - **Metadata:** {_result.get('metadata', _result.get('payload', {}))}
                                """
                            )
                        )

                    cell9_out = mo.vstack(
                        [
                            mo.md(f"## 🔍 Search Test Results ({db_type.upper()})"),
                            mo.md(f"**Query:** {_search_query}"),
                            mo.md(f"**Top-K:** {_top_k}"),
                            mo.md(f"**Results Found:** {len(_results)}"),
                            mo.md(f"**Execution Time:** {execution_time:.3f}s"),
                            mo.md("### 📋 Search Results"),
                            mo.md(
                                "\n".join(result_displays)
                                if result_displays
                                else "No _results found"
                            ),
                        ]
                    )

            elif operation == "Get Statistics":
                # Get database statistics
                _stats = db_manager.execute_operation("get_stats")

                cell9_out = mo.vstack(
                    [
                        mo.md(f"## 📊 Database Statistics ({db_type.upper()})"),
                        mo.md("### 📈 Database Info"),
                        mo.md(f"```json\n{json.dumps(_stats, indent=2)}\n```"),
                    ]
                )

            elif operation == "Performance Test":
                # Run comprehensive performance test
                _performance_results = {}

                # Test vector addition
                _test_vectors = [
                    [random.random() for _ in range(vector_dim)]
                    for _ in range(_num_vectors)
                ]
                _test_metadata = [
                    {"test": True, "index": _i} for _i in range(_num_vectors)
                ]

                _start_time = time.time()
                add_result = db_manager.execute_operation(
                    "add_vectors", _test_vectors, _test_metadata
                )
                _add_time = time.time() - _start_time

                # Test search
                _query_vector = [random.random() for _ in range(vector_dim)]
                _start_time = time.time()
                _search_results = db_manager.execute_operation(
                    "search", _query_vector, _top_k
                )
                _search_time = time.time() - _start_time

                _performance_results = {
                    "add_vectors": {
                        "time": _add_time,
                        "throughput": _num_vectors / _add_time,
                        "success": add_result,
                    },
                    "search": {
                        "time": _search_time,
                        "results_found": len(_search_results),
                        "latency_per_result": _search_time
                        / max(len(_search_results), 1),
                    },
                }

                cell9_out = mo.vstack(
                    [
                        mo.md(f"## ⚡ Performance Test Results ({db_type.upper()})"),
                        mo.md("### 📊 Add Vectors Performance"),
                        mo.md(
                            f"- **Time:** {_performance_results['add_vectors']['time']:.3f}s"
                        ),
                        mo.md(
                            f"- **Throughput:** {_performance_results['add_vectors']['throughput']:.1f} vectors/s"
                        ),
                        mo.md("### 🔍 Search Performance"),
                        mo.md(
                            f"- **Time:** {_performance_results['search']['time']:.3f}s"
                        ),
                        mo.md(
                            f"- **Results:** {_performance_results['search']['results_found']}"
                        ),
                        mo.md(
                            f"- **Latency per Result:** {_performance_results['search']['latency_per_result']:.4f}s"
                        ),
                    ]
                )

        except Exception as e:
            cell9_out = mo.md(f"❌ **Database Test Error:** {str(e)}")
    else:
        cell9_out = mo.md(
            "*Configure test parameters and click 'Run Database Test' to test the databases*"
        )

    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, cleandoc, db_manager, mo, output):
    if available_providers and db_manager:
        cell10_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 7: Performance Comparison Dashboard

                Let's compare performance across all vector databases:
                """
            )
        )

        # Performance comparison button
        run_comparison = mo.ui.run_button(label="⚡ Run Performance Comparison")

        cell10_content = mo.vstack(
            [
                mo.md("### ⚡ Multi-Database Performance Comparison"),
                mo.md("Compare performance across FAISS, ChromaDB, and Qdrant:"),
                run_comparison,
            ]
        )
    else:
        cell10_desc = mo.md("")
        run_comparison = None
        cell10_content = mo.md("")

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return (run_comparison,)


@app.cell
def _(
    available_providers,
    cleandoc,
    db_manager,
    mo,
    output,
    random,
    run_comparison,
    time,
):
    if available_providers and run_comparison.value and db_manager:
        try:
            # Performance comparison parameters
            test_vector_count = 20
            test_dimension = 128
            search_iterations = 5

            # Generate test data
            _test_vectors = [
                [random.random() for _ in range(test_dimension)]
                for _ in range(test_vector_count)
            ]
            _test_metadata = [
                {"category": f"cat_{_i%3}", "score": random.random()}
                for _i in range(test_vector_count)
            ]
            _test_ids = [f"perf_test_{_i}" for _i in range(test_vector_count)]

            comparison_results = {}

            # Test each database
            for _db_name in ["faiss", "chromadb", "qdrant"]:
                try:
                    db_manager.set_active_database(_db_name)

                    # Test vector addition
                    _start_time = time.time()
                    add_success = db_manager.execute_operation(
                        "add_vectors", _test_vectors, _test_metadata, _test_ids
                    )
                    _add_time = time.time() - _start_time

                    # Test search performance
                    search_times = []
                    search_results_counts = []

                    for _ in range(search_iterations):
                        _query_vector = [random.random() for _ in range(test_dimension)]
                        _start_time = time.time()
                        _search_results = db_manager.execute_operation(
                            "search", _query_vector, 5
                        )
                        _search_time = time.time() - _start_time

                        search_times.append(_search_time)
                        search_results_counts.append(len(_search_results))

                    avg_search_time = sum(search_times) / len(search_times)
                    avg_results_count = sum(search_results_counts) / len(
                        search_results_counts
                    )

                    # Get database _stats
                    _stats = db_manager.execute_operation("get_stats")

                    comparison_results[_db_name] = {
                        "_add_time": _add_time,
                        "add_throughput": test_vector_count / _add_time,
                        "avg_search_time": avg_search_time,
                        "avg_results_count": avg_results_count,
                        "search_qps": 1 / avg_search_time,
                        "total_vectors": _stats.get("total_vectors", 0),
                        "add_success": add_success,
                    }

                except Exception as e:
                    comparison_results[_db_name] = {"error": str(e)}

            # Create comparison table
            comparison_table = []
            for _db_name, _results in comparison_results.items():
                if "error" not in _results:
                    comparison_table.append(
                        {
                            "Database": _db_name.upper(),
                            "Add Time (s)": f"{_results['_add_time']:.3f}",
                            "Add Throughput (vec/s)": f"{_results['add_throughput']:.1f}",
                            "Avg Search Time (s)": f"{_results['avg_search_time']:.4f}",
                            "Search QPS": f"{_results['search_qps']:.1f}",
                            "Total Vectors": _results["total_vectors"],
                            "Status": (
                                "✅ Success" if _results["add_success"] else "❌ Failed"
                            ),
                        }
                    )
                else:
                    comparison_table.append(
                        {
                            "Database": _db_name.upper(),
                            "Add Time (s)": "Error",
                            "Add Throughput (vec/s)": "Error",
                            "Avg Search Time (s)": "Error",
                            "Search QPS": "Error",
                            "Total Vectors": "Error",
                            "Status": f"❌ {_results['error']}",
                        }
                    )

            # Performance insights
            successful_results = {
                k: v for k, v in comparison_results.items() if "error" not in v
            }

            insights = []
            if successful_results:
                # Find fastest for different operations
                fastest_add = min(
                    successful_results.items(), key=lambda x: x[1]["_add_time"]
                )
                fastest_search = min(
                    successful_results.items(), key=lambda x: x[1]["avg_search_time"]
                )

                insights.extend(
                    [
                        f"🏆 **Fastest Vector Addition:** {fastest_add[0].upper()} ({fastest_add[1]['_add_time']:.3f}s)",
                        f"🔍 **Fastest Search:** {fastest_search[0].upper()} ({fastest_search[1]['avg_search_time']:.4f}s)",
                        f"📊 **Test Configuration:** {test_vector_count} vectors, {test_dimension}D, {search_iterations} search iterations",
                    ]
                )

            cell11_out = mo.vstack(
                [
                    mo.md("## ⚡ Performance Comparison Results"),
                    mo.md("### 📊 Performance Metrics"),
                    mo.ui.table(comparison_table),
                    mo.md("### 💡 Performance Insights"),
                    mo.md(
                        "\n".join(insights)
                        if insights
                        else "No successful tests to analyze"
                    ),
                    mo.md(
                        cleandoc(
                            """
                            ### 🎯 Database Selection Guidelines

                            **FAISS:**  
                            - ✅ Best for: High-performance similarity search, large-scale datasets  
                            - ✅ Strengths: Fastest search, memory efficient, battle-tested  
                            - ❌ Limitations: No built-in persistence, limited metadata support  

                            **ChromaDB:**  
                            - ✅ Best for: Document-centric applications, rich metadata  
                            - ✅ Strengths: Easy integration, full-text search, good for prototyping  
                            - ❌ Limitations: Slower than FAISS, newer ecosystem  

                            **Qdrant:**  
                            - ✅ Best for: Production deployments, advanced filtering  
                            - ✅ Strengths: Rust performance, rich filtering, cloud-ready  
                            - ❌ Limitations: More complex setup, resource intensive  

                            ### 🏭 Production Recommendations

                            - **High Performance + Simple Use Case:** FAISS  
                            - **Rich Metadata + Easy Integration:** ChromaDB   
                            - **Production Scale + Advanced Features:** Qdrant  
                            - **Hybrid Approach:** Use different databases for different use cases  
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell11_out = mo.md(f"❌ **Performance Comparison Error:** {str(e)}")
    else:
        cell11_out = mo.md(
            "*Click 'Run Performance Comparison' to benchmark all vector databases*"
        )

    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell12_out = mo.md(
        cleandoc(
            """
            ## 🎓 Vector Database Setup Module Complete!

            ### 🏆 What You've Mastered

            ✅ **Unified Database Interface** - Common API across multiple vector databases  
            ✅ **FAISS Integration** - High-performance similarity search implementation  
            ✅ **ChromaDB Integration** - Document-centric vector database with metadata  
            ✅ **Qdrant Integration** - Production-ready vector search engine  
            ✅ **Performance Comparison** - Systematic benchmarking and optimization  
            ✅ **Database Management** - Centralized management and switching capabilities  

            ### 🛠️ Key Components Built

            1. **VectorDatabaseInterface**  
                - Abstract base class for all vector databases  
                - Standardized operations: add, search, delete, _stats  
                - Consistent error handling and response formats  

            2. **VectorDatabaseManager**  
                - Multi-database registration and management  
                - Performance tracking and metrics collection  
                - Seamless database switching and comparison  

            3. **Database Implementations**  
            - **FAISS**: High-performance similarity search  
            - **ChromaDB**: Document-centric with rich metadata  
            - **Qdrant**: Production-ready with advanced filtering  

            4. **Testing and Benchmarking**  
                - Comprehensive testing interface  
                - Performance comparison dashboard  
                - Real-world usage simulation  

            ### 🎯 Skills Developed

            - **Database Architecture** - Designing unified interfaces for multiple backends
            - **Performance Optimization** - Benchmarking and comparing vector databases
            - **Production Deployment** - Understanding trade-offs and selection criteria
            - **System Integration** - Connecting different vector database technologies
            - **Monitoring and Analytics** - Tracking performance and usage metrics

            ### 🚀 Ready for Retrieval Optimization?

            You now understand comprehensive vector database management! Time to optimize retrieval:

            **Next Module:**
            ```bash
            uv run marimo run 03-retrieval-rag/advanced_rag_patterns.py
            ```

            **Coming Up:**
            - Custom retrievers and ranking algorithms
            - Retrieval evaluation metrics and benchmarking
            - Interactive parameter tuning for retrieval components
            - Advanced retrieval strategies and techniques

            ### 💡 Advanced Practice Challenges

            Before moving on, try extending the database system:

            1. **Production Integration**
                - Connect to real FAISS, ChromaDB, or Qdrant instances
                - Implement proper embedding models (OpenAI, sentence-transformers)
                - Add connection pooling and error recovery

            2. **Advanced Features**
                - Implement hybrid search (vector + keyword)
                - Add batch operations for better performance
                - Create database migration and backup tools

            3. **Monitoring and Observability**
                - Add comprehensive logging and metrics
                - Implement health checks and alerting
                - Create performance dashboards

            4. **Scalability Enhancements**
                - Implement sharding and distributed search
                - Add caching layers for frequently accessed vectors
                - Optimize memory usage and garbage collection

            ### 🏭 Production Deployment Checklist

            When deploying vector databases in production:  
            - [ ] Choose appropriate database based on use case requirements  
            - [ ] Implement proper connection management and pooling  
            - [ ] Set up monitoring, logging, and alerting  
            - [ ] Plan for data backup and disaster recovery  
            - [ ] Implement security measures and access controls  
            - [ ] Design for horizontal scaling and load distribution  
            - [ ] Establish performance baselines and SLAs  

            Master these vector database patterns and you can build scalable, high-performance retrieval systems for any application!
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell12_out)
    return


if __name__ == "__main__":
    app.run()
