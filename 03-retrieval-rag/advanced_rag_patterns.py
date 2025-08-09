# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        Any,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        re,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🚀 Advanced RAG Patterns and Techniques

            **Duration:** 120-150 minutes  
            **Prerequisites:** Completed Basic RAG Implementation  
            **Difficulty:** Advanced  

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Master multi-hop reasoning in RAG systems  
            - ✅ Implement query decomposition and planning  
            - ✅ Build hierarchical and fusion retrieval systems  
            - ✅ Create adaptive RAG with dynamic strategies  
            - ✅ Implement advanced re-ranking and filtering  

            ## 🧩 Advanced RAG Patterns

            **Beyond Basic RAG:**  
            - **Multi-Hop Reasoning** - Chain multiple retrieval-generation cycles  
            - **Query Decomposition** - Break complex queries into sub-questions  
            - **Hierarchical Retrieval** - Multi-level document organization  
            - **Fusion RAG** - Combine multiple retrieval strategies  
            - **Adaptive RAG** - Dynamic strategy selection based on query type  

            Let's build sophisticated RAG systems!
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
                ## ✅ Advanced RAG Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  

                Ready to build advanced RAG patterns!
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
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🔗 Step 1: Multi-Hop Reasoning RAG

                Multi-hop reasoning allows RAG systems to chain multiple retrieval-generation cycles to answer complex questions that require information from multiple sources.
                """
            )
        )

        # Multi-Hop RAG Implementation
        class MultiHopRAG(dspy.Module):
            """Multi-hop reasoning RAG system."""

            def __init__(self, retriever, max_hops=3):
                super().__init__()
                self.retriever = retriever
                self.max_hops = max_hops

                # Define signatures for different stages
                self.question_decomposer = dspy.ChainOfThought(
                    "question -> sub_questions"
                )
                self.answer_synthesizer = dspy.ChainOfThought(
                    "context, question -> answer"
                )
                self.hop_planner = dspy.ChainOfThought(
                    "question, previous_context -> next_query"
                )

            def forward(self, question):
                """Execute multi-hop reasoning."""
                # Step 1: Decompose the question
                decomposition = self.question_decomposer(question=question)
                sub_questions = self._parse_sub_questions(decomposition.sub_questions)

                # Step 2: Multi-hop retrieval and reasoning
                all_context = []
                current_question = question

                for hop in range(min(len(sub_questions), self.max_hops)):
                    # Retrieve relevant documents
                    retrieved_docs = self.retriever(current_question, k=5)
                    hop_context = "\n".join([doc["text"] for doc in retrieved_docs])
                    all_context.append(f"Hop {hop + 1}: {hop_context}")

                    # Plan next query if not the last hop
                    if hop < len(sub_questions) - 1:
                        next_query_result = self.hop_planner(
                            question=sub_questions[hop + 1],
                            previous_context=hop_context,
                        )
                        current_question = next_query_result.next_query

                # Step 3: Synthesize final answer
                combined_context = "\n\n".join(all_context)
                final_answer = self.answer_synthesizer(
                    context=combined_context, question=question
                )

                return dspy.Prediction(
                    answer=final_answer.answer,
                    context=combined_context,
                    hops=len(all_context),
                    sub_questions=sub_questions,
                )

            def _parse_sub_questions(self, sub_questions_text):
                """Parse sub-questions from text."""
                lines = sub_questions_text.strip().split("\n")
                questions = []
                for line in lines:
                    line = line.strip()
                    if line and (
                        line.startswith("-")
                        or line.startswith("1.")
                        or line.startswith("•")
                    ):
                        # Remove bullet points and numbering
                        clean_question = line.lstrip("- 1234567890.• ").strip()
                        if clean_question:
                            questions.append(clean_question)
                return questions if questions else [sub_questions_text]

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🔗 Multi-Hop RAG System Created

                **Key Features:**  
                - **Question Decomposition** - Breaks complex questions into sub-questions  
                - **Iterative Retrieval** - Multiple retrieval cycles for comprehensive context  
                - **Hop Planning** - Intelligent query refinement between hops  
                - **Context Synthesis** - Combines information from all hops  

                The multi-hop system can handle complex questions requiring multiple information sources!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        MultiHopRAG = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (MultiHopRAG,)


@app.cell
def _(MultiHopRAG, available_providers, cleandoc, dspy, mo, output):
    if available_providers and MultiHopRAG:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🧩 Step 2: Query Decomposition and Planning

                Advanced query decomposition breaks down complex questions into manageable sub-tasks with intelligent planning.
                """
            )
        )

        # Query Decomposition System
        class QueryDecomposer(dspy.Module):
            """Advanced query decomposition with planning."""

            def __init__(self):
                super().__init__()
                self.complexity_analyzer = dspy.ChainOfThought(
                    "question -> complexity_score, reasoning"
                )
                self.decomposer = dspy.ChainOfThought(
                    "question, complexity -> sub_queries, execution_plan"
                )
                self.dependency_mapper = dspy.ChainOfThought(
                    "sub_queries -> dependencies, execution_order"
                )

            def forward(self, question):
                """Decompose query with execution planning."""
                # Analyze complexity
                complexity = self.complexity_analyzer(question=question)

                # Decompose based on complexity
                decomposition = self.decomposer(
                    question=question, complexity=complexity.complexity_score
                )

                # Map dependencies
                dependencies = self.dependency_mapper(
                    sub_queries=decomposition.sub_queries
                )

                return dspy.Prediction(
                    sub_queries=self._parse_queries(decomposition.sub_queries),
                    execution_plan=decomposition.execution_plan,
                    dependencies=dependencies.dependencies,
                    execution_order=dependencies.execution_order,
                    complexity_score=complexity.complexity_score,
                )

            def _parse_queries(self, queries_text):
                """Parse sub-queries from text."""
                lines = queries_text.strip().split("\n")
                queries = []
                for line in lines:
                    line = line.strip()
                    if line and any(
                        line.startswith(prefix)
                        for prefix in ["-", "1.", "2.", "3.", "4.", "5.", "•"]
                    ):
                        clean_query = line.lstrip("- 1234567890.• ").strip()
                        if clean_query:
                            queries.append(clean_query)
                return queries if queries else [queries_text]

        class PlanExecutor(dspy.Module):
            """Execute decomposed queries according to plan."""

            def __init__(self, retriever):
                super().__init__()
                self.retriever = retriever
                self.query_executor = dspy.ChainOfThought("query, context -> result")
                self.result_combiner = dspy.ChainOfThought(
                    "results, original_question -> final_answer"
                )

            def forward(self, decomposition, original_question):
                """Execute the decomposed query plan."""
                results = {}
                execution_context = ""

                # Execute queries in planned order
                for i, query in enumerate(decomposition.sub_queries):
                    # Retrieve relevant documents
                    retrieved_docs = self.retriever(query, k=3)
                    query_context = "\n".join([doc["text"] for doc in retrieved_docs])

                    # Execute query with accumulated context
                    result = self.query_executor(
                        query=query, context=f"{execution_context}\n{query_context}"
                    )

                    results[f"query_{i+1}"] = {
                        "query": query,
                        "result": result.result,
                        "context": query_context,
                    }

                    # Update execution context
                    execution_context += f"\nQuery {i+1} Result: {result.result}"

                # Combine all results
                combined_result = self.result_combiner(
                    results=str(results), original_question=original_question
                )

                return dspy.Prediction(
                    final_answer=combined_result.final_answer,
                    sub_results=results,
                    execution_trace=execution_context,
                )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🧩 Query Decomposition System Created

                **Components:**  
                - **Complexity Analyzer** - Assesses question difficulty and scope  
                - **Smart Decomposer** - Breaks questions into logical sub-queries  
                - **Dependency Mapper** - Identifies query dependencies and execution order  
                - **Plan Executor** - Executes queries according to the optimal plan  

                The system can handle complex, multi-faceted questions with intelligent planning!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        QueryDecomposer = None
        PlanExecutor = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (QueryDecomposer,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 3: Hierarchical Retrieval System

                Hierarchical retrieval organizes documents at multiple levels for more precise and contextual information retrieval.
                """
            )
        )

        # Hierarchical Retrieval Implementation
        class HierarchicalRetriever(dspy.Module):
            """Multi-level hierarchical document retrieval."""

            def __init__(self, document_hierarchy):
                super().__init__()
                self.hierarchy = document_hierarchy
                self.level_selectors = {
                    "domain": dspy.ChainOfThought("question -> relevant_domains"),
                    "category": dspy.ChainOfThought(
                        "question, domain -> relevant_categories"
                    ),
                    "document": dspy.ChainOfThought(
                        "question, category -> relevant_documents"
                    ),
                }
                self.relevance_scorer = dspy.ChainOfThought(
                    "question, document -> relevance_score, reasoning"
                )

            def forward(self, question, max_docs=10):
                """Perform hierarchical retrieval."""
                # Level 1: Domain Selection
                domain_selection = self.level_selectors["domain"](question=question)
                relevant_domains = self._parse_domains(
                    domain_selection.relevant_domains
                )

                # Level 2: Category Selection within domains
                relevant_categories = []
                for domain in relevant_domains:
                    if domain in self.hierarchy:
                        category_selection = self.level_selectors["category"](
                            question=question, domain=domain
                        )
                        categories = self._parse_categories(
                            category_selection.relevant_categories
                        )
                        relevant_categories.extend(
                            [(domain, cat) for cat in categories]
                        )

                # Level 3: Document Selection within categories
                candidate_documents = []
                for domain, category in relevant_categories:
                    if domain in self.hierarchy and category in self.hierarchy[domain]:
                        doc_selection = self.level_selectors["document"](
                            question=question, category=f"{domain}/{category}"
                        )
                        docs = self.hierarchy[domain][category]
                        candidate_documents.extend(docs)

                # Level 4: Relevance Scoring and Ranking
                scored_documents = []
                for doc in candidate_documents[
                    : max_docs * 2
                ]:  # Score more than needed
                    score_result = self.relevance_scorer(
                        question=question, document=doc["text"]
                    )
                    try:
                        score = float(score_result.relevance_score.split()[0])
                    except Exception as _:
                        score = 0.5  # Default score

                    scored_documents.append(
                        {
                            **doc,
                            "relevance_score": score,
                            "reasoning": score_result.reasoning,
                        }
                    )

                # Sort by relevance and return top documents
                scored_documents.sort(key=lambda x: x["relevance_score"], reverse=True)

                return dspy.Prediction(
                    documents=scored_documents[:max_docs],
                    domains_searched=relevant_domains,
                    categories_searched=relevant_categories,
                    total_candidates=len(candidate_documents),
                )

            def _parse_domains(self, domains_text):
                """Parse domain names from text."""
                domains = []
                for line in domains_text.strip().split("\n"):
                    line = line.strip().lstrip("- •").strip()
                    if line and line in self.hierarchy:
                        domains.append(line)
                return domains if domains else list(self.hierarchy.keys())[:2]

            def _parse_categories(self, categories_text):
                """Parse category names from text."""
                categories = []
                for line in categories_text.strip().split("\n"):
                    line = line.strip().lstrip("- •").strip()
                    if line:
                        categories.append(line)
                return categories if categories else ["general"]

        # Sample hierarchical document structure
        sample_hierarchy = {
            "technology": {
                "ai_ml": [
                    {
                        "id": "tech_ai_1",
                        "text": "Machine learning algorithms for pattern recognition...",
                        "title": "ML Patterns",
                    },
                    {
                        "id": "tech_ai_2",
                        "text": "Deep learning architectures and neural networks...",
                        "title": "Deep Learning",
                    },
                ],
                "software": [
                    {
                        "id": "tech_sw_1",
                        "text": "Software architecture patterns and best practices...",
                        "title": "Architecture",
                    },
                    {
                        "id": "tech_sw_2",
                        "text": "Database design and optimization techniques...",
                        "title": "Databases",
                    },
                ],
            },
            "science": {
                "physics": [
                    {
                        "id": "sci_phy_1",
                        "text": "Quantum mechanics principles and applications...",
                        "title": "Quantum Physics",
                    },
                    {
                        "id": "sci_phy_2",
                        "text": "Thermodynamics and energy conservation laws...",
                        "title": "Thermodynamics",
                    },
                ],
                "biology": [
                    {
                        "id": "sci_bio_1",
                        "text": "Cellular biology and molecular processes...",
                        "title": "Cell Biology",
                    },
                    {
                        "id": "sci_bio_2",
                        "text": "Genetics and heredity mechanisms...",
                        "title": "Genetics",
                    },
                ],
            },
        }

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Hierarchical Retrieval System Created

                **Hierarchy Levels:**  
                1. **Domain Level** - High-level subject areas (technology, science, etc.)  
                2. **Category Level** - Specific topics within domains (AI/ML, physics, etc.)  
                3. **Document Level** - Individual documents within categories  
                4. **Relevance Scoring** - Fine-grained relevance assessment  

                **Benefits:**  
                - **Precision** - More targeted document selection  
                - **Scalability** - Efficient search in large document collections  
                - **Context Awareness** - Maintains topical coherence  
                - **Explainability** - Clear retrieval path through hierarchy  

                The hierarchical system provides structured, efficient document retrieval!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        HierarchicalRetriever = None
        sample_hierarchy = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return HierarchicalRetriever, sample_hierarchy


@app.cell
def _(Any, available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🔀 Step 4: Fusion RAG - Multiple Retrieval Strategies

                Fusion RAG combines multiple retrieval strategies to maximize information coverage and accuracy.
                """
            )
        )

        # Fusion RAG Implementation
        class FusionRAG(dspy.Module):
            """Multi-strategy fusion RAG system."""

            def __init__(self, retrievers: dict[str, Any]):
                super().__init__()
                self.retrievers = retrievers
                self.strategy_selector = dspy.ChainOfThought(
                    "question -> best_strategies, reasoning"
                )
                self.result_ranker = dspy.ChainOfThought(
                    "question, results -> ranked_results, fusion_reasoning"
                )
                self.answer_generator = dspy.ChainOfThought(
                    "question, fused_context -> answer"
                )

            def forward(
                self, question, max_docs_per_strategy=5, fusion_method="weighted_rank"
            ):
                """Execute fusion retrieval and generation."""
                # Step 1: Select optimal retrieval strategies
                strategy_selection = self.strategy_selector(question=question)
                selected_strategies = self._parse_strategies(
                    strategy_selection.best_strategies
                )

                # Step 2: Execute multiple retrieval strategies
                all_results = {}
                for strategy_name in selected_strategies:
                    if strategy_name in self.retrievers:
                        try:
                            results = self.retrievers[strategy_name](
                                question, k=max_docs_per_strategy
                            )
                            all_results[strategy_name] = results
                        except Exception as e:
                            print(f"Strategy {strategy_name} failed: {e}")
                            all_results[strategy_name] = []

                # Step 3: Fusion and ranking
                fused_results = self._fuse_results(all_results, fusion_method)

                # Step 4: Re-rank with LLM
                ranking_result = self.result_ranker(
                    question=question,
                    results=str(
                        [doc["text"][:200] + "..." for doc in fused_results[:10]]
                    ),
                )

                # Step 5: Generate final answer
                fused_context = "\n\n".join([doc["text"] for doc in fused_results[:8]])
                final_answer = self.answer_generator(
                    question=question, fused_context=fused_context
                )

                return dspy.Prediction(
                    answer=final_answer.answer,
                    fused_documents=fused_results,
                    strategies_used=selected_strategies,
                    fusion_method=fusion_method,
                    strategy_results=all_results,
                )

            def _parse_strategies(self, strategies_text):
                """Parse strategy names from text."""
                available_strategies = list(self.retrievers.keys())
                strategies = []

                for line in strategies_text.strip().split("\n"):
                    line = line.strip().lstrip("- •1234567890.").strip().lower()
                    for available in available_strategies:
                        if available.lower() in line or line in available.lower():
                            if available not in strategies:
                                strategies.append(available)

                return strategies if strategies else available_strategies[:2]

            def _fuse_results(
                self, all_results: dict[str, list], method: str
            ) -> list[dict]:
                """Fuse results from multiple strategies."""
                if method == "weighted_rank":
                    return self._weighted_rank_fusion(all_results)
                elif method == "reciprocal_rank":
                    return self._reciprocal_rank_fusion(all_results)
                else:
                    return self._simple_concatenation(all_results)

            def _weighted_rank_fusion(self, all_results: dict[str, list]) -> list[dict]:
                """Weighted rank fusion of results."""
                strategy_weights = {"semantic": 0.4, "keyword": 0.3, "hybrid": 0.3}

                doc_scores = {}
                for strategy, results in all_results.items():
                    weight = strategy_weights.get(strategy, 0.2)
                    for i, doc in enumerate(results):
                        doc_id = doc.get("id", f"{strategy}_{i}")
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = {"doc": doc, "score": 0}

                        # Higher rank = lower index, so invert
                        rank_score = weight * (1.0 / (i + 1))
                        doc_scores[doc_id]["score"] += rank_score

                # Sort by fused score
                sorted_docs = sorted(
                    doc_scores.values(), key=lambda x: x["score"], reverse=True
                )
                return [item["doc"] for item in sorted_docs]

            def _reciprocal_rank_fusion(
                self, all_results: dict[str, list]
            ) -> list[dict]:
                """Reciprocal rank fusion (RRF)."""
                k = 60  # RRF parameter
                doc_scores = {}

                for strategy, results in all_results.items():
                    for i, doc in enumerate(results):
                        doc_id = doc.get("id", f"{strategy}_{i}")
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = {"doc": doc, "score": 0}

                        rrf_score = 1.0 / (k + i + 1)
                        doc_scores[doc_id]["score"] += rrf_score

                sorted_docs = sorted(
                    doc_scores.values(), key=lambda x: x["score"], reverse=True
                )
                return [item["doc"] for item in sorted_docs]

            def _simple_concatenation(self, all_results: dict[str, list]) -> list[dict]:
                """Simple concatenation with deduplication."""
                seen_ids = set()
                fused = []

                for strategy, results in all_results.items():
                    for doc in results:
                        doc_id = doc.get("id", doc.get("text", "")[:50])
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            fused.append(doc)

                return fused

        cell6_content = mo.md(
            cleandoc(
                """
                ### 🔀 Fusion RAG System Created

                **Fusion Strategies:**  
                - **Weighted Rank Fusion** - Combines results with strategy-specific weights  
                - **Reciprocal Rank Fusion (RRF)** - Standard fusion algorithm for search  
                - **Simple Concatenation** - Basic deduplication and combination  

                **Multi-Strategy Support:**  
                - **Semantic Search** - Vector similarity-based retrieval  
                - **Keyword Search** - Traditional text matching  
                - **Hybrid Search** - Combination of semantic and keyword  
                - **Custom Strategies** - Extensible architecture for new methods  

                **Benefits:**  
                - **Comprehensive Coverage** - Multiple retrieval perspectives  
                - **Robustness** - Fallback when individual strategies fail  
                - **Quality Improvement** - Best results from each strategy  
                - **Adaptability** - Strategy selection based on query type  

                The fusion system maximizes retrieval quality through strategy diversity!
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        FusionRAG = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (FusionRAG,)


@app.cell
def _(Any, available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Step 5: Adaptive RAG - Dynamic Strategy Selection

                Adaptive RAG automatically selects the best retrieval and generation strategy based on query characteristics.
                """
            )
        )

        # Adaptive RAG Implementation
        class AdaptiveRAG(dspy.Module):
            """Adaptive RAG with dynamic strategy selection."""

            def __init__(self, strategy_modules: dict[str, Any]):
                super().__init__()
                self.strategies = strategy_modules
                self.query_analyzer = dspy.ChainOfThought(
                    "question -> query_type, complexity, domain, reasoning"
                )
                self.strategy_selector = dspy.ChainOfThought(
                    "query_analysis -> best_strategy, confidence, reasoning"
                )
                self.performance_tracker = {}

            def forward(self, question):
                """Execute adaptive RAG with dynamic strategy selection."""
                # Step 1: Analyze the query
                analysis = self.query_analyzer(question=question)

                # Step 2: Select optimal strategy
                strategy_selection = self.strategy_selector(
                    query_analysis=str(analysis)
                )
                selected_strategy = self._parse_strategy(
                    strategy_selection.best_strategy
                )

                # Step 3: Execute selected strategy
                if selected_strategy in self.strategies:
                    try:
                        result = self.strategies[selected_strategy](question)

                        # Track performance
                        self._update_performance(selected_strategy, True)

                        return dspy.Prediction(
                            answer=(
                                result.answer
                                if hasattr(result, "answer")
                                else str(result)
                            ),
                            strategy_used=selected_strategy,
                            query_analysis=analysis,
                            confidence=strategy_selection.confidence,
                            reasoning=strategy_selection.reasoning,
                            performance_stats=self.performance_tracker.get(
                                selected_strategy, {}
                            ),
                        )
                    except Exception as e:
                        # Fallback to default strategy
                        self._update_performance(selected_strategy, False)
                        return self._fallback_execution(question, str(e))
                else:
                    return self._fallback_execution(
                        question, f"Strategy {selected_strategy} not available"
                    )

            def _parse_strategy(self, strategy_text):
                """Parse strategy name from text."""
                strategy_text = strategy_text.lower().strip()
                available_strategies = list(self.strategies.keys())

                for strategy in available_strategies:
                    if (
                        strategy.lower() in strategy_text
                        or strategy_text in strategy.lower()
                    ):
                        return strategy

                # Default to first available strategy
                return available_strategies[0] if available_strategies else "basic"

            def _update_performance(self, strategy: str, success: bool):
                """Update performance tracking for strategies."""
                if strategy not in self.performance_tracker:
                    self.performance_tracker[strategy] = {
                        "total_calls": 0,
                        "successful_calls": 0,
                        "success_rate": 0.0,
                    }

                stats = self.performance_tracker[strategy]
                stats["total_calls"] += 1
                if success:
                    stats["successful_calls"] += 1
                stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]

            def _fallback_execution(self, question, error_msg):
                """Execute fallback strategy when primary fails."""
                fallback_strategies = ["basic", "simple", "default"]

                for fallback in fallback_strategies:
                    if fallback in self.strategies:
                        try:
                            result = self.strategies[fallback](question)
                            return dspy.Prediction(
                                answer=(
                                    result.answer
                                    if hasattr(result, "answer")
                                    else str(result)
                                ),
                                strategy_used=f"fallback_{fallback}",
                                error=error_msg,
                                fallback_used=True,
                            )
                        except Exception as _:
                            continue

                # Ultimate fallback
                return dspy.Prediction(
                    answer="I apologize, but I'm unable to process this query at the moment.",
                    strategy_used="error_fallback",
                    error=error_msg,
                    fallback_used=True,
                )

        class QueryTypeClassifier(dspy.Module):
            """Classify queries to determine optimal RAG strategy."""

            def __init__(self):
                super().__init__()
                self.classifier = dspy.ChainOfThought(
                    "question -> category, confidence, features"
                )

            def forward(self, question):
                """Classify query type for strategy selection."""
                classification = self.classifier(question=question)

                # Parse classification results
                category = self._parse_category(classification.category)
                features = self._parse_features(classification.features)

                return dspy.Prediction(
                    category=category,
                    confidence=classification.confidence,
                    features=features,
                    recommendation=self._get_strategy_recommendation(
                        category, features
                    ),
                )

            def _parse_category(self, category_text):
                """Parse query category."""
                category_text = category_text.lower().strip()
                categories = {
                    "factual": ["fact", "what", "when", "where", "who"],
                    "analytical": ["analyze", "compare", "evaluate", "assess"],
                    "creative": ["create", "generate", "design", "imagine"],
                    "complex": ["complex", "multi-step", "reasoning", "chain"],
                }

                for cat, keywords in categories.items():
                    if any(keyword in category_text for keyword in keywords):
                        return cat

                return "general"

            def _parse_features(self, features_text):
                """Parse query features."""
                features = []
                feature_keywords = {
                    "multi_hop": ["multiple", "chain", "step", "sequence"],
                    "domain_specific": ["technical", "specialized", "domain"],
                    "temporal": ["time", "recent", "historical", "timeline"],
                    "comparative": ["compare", "versus", "difference", "similar"],
                }

                features_text = features_text.lower()
                for feature, keywords in feature_keywords.items():
                    if any(keyword in features_text for keyword in keywords):
                        features.append(feature)

                return features

            def _get_strategy_recommendation(self, category, features):
                """Recommend strategy based on classification."""
                if "multi_hop" in features:
                    return "multi_hop"
                elif "comparative" in features:
                    return "fusion"
                elif category == "complex":
                    return "hierarchical"
                elif "domain_specific" in features:
                    return "hierarchical"
                else:
                    return "basic"

        cell7_content = mo.md(
            cleandoc(
                """
                ### 🎯 Adaptive RAG System Created

                **Adaptive Components:**  
                - **Query Analyzer** - Analyzes query type, complexity, and domain  
                - **Strategy Selector** - Chooses optimal RAG strategy dynamically  
                - **Performance Tracker** - Monitors strategy success rates  
                - **Fallback System** - Handles failures gracefully  

                **Query Classification:**  
                - **Factual** - Direct fact retrieval queries  
                - **Analytical** - Comparison and analysis queries  
                - **Creative** - Generation and synthesis queries  
                - **Complex** - Multi-step reasoning queries  

                **Strategy Mapping:**  
                - **Multi-hop** → Complex, chained reasoning queries  
                - **Fusion** → Comparative and comprehensive queries  
                - **Hierarchical** → Domain-specific, structured queries  
                - **Basic** → Simple, direct queries  

                **Benefits:**  
                - **Intelligent Selection** - Optimal strategy for each query type  
                - **Performance Optimization** - Learns from success/failure patterns  
                - **Robustness** - Automatic fallback mechanisms  
                - **Adaptability** - Improves over time with usage  

                The adaptive system provides intelligent, self-optimizing RAG!
                """
            )
        )
    else:
        cell7_desc = mo.md("")
        AdaptiveRAG = None
        QueryTypeClassifier = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return (AdaptiveRAG,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output, re):
    if available_providers:
        cell8_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 6: Advanced Re-ranking and Filtering

                Advanced re-ranking improves retrieval quality through sophisticated scoring and filtering mechanisms.
                """
            )
        )

        # Advanced Re-ranking System
        class AdvancedReranker(dspy.Module):
            """Multi-criteria document re-ranking system."""

            def __init__(self):
                super().__init__()
                self.relevance_scorer = dspy.ChainOfThought(
                    "question, document -> relevance_score, reasoning"
                )
                self.quality_assessor = dspy.ChainOfThought(
                    "document -> quality_score, quality_factors"
                )
                self.diversity_calculator = dspy.ChainOfThought(
                    "documents -> diversity_score, unique_aspects"
                )
                self.final_ranker = dspy.ChainOfThought(
                    "scores, question -> final_ranking, ranking_reasoning"
                )

            def forward(self, question, documents, top_k=10):
                """Re-rank documents using multiple criteria."""
                if not documents:
                    return dspy.Prediction(ranked_documents=[], ranking_scores={})

                # Step 1: Score each document on multiple criteria
                scored_documents = []
                for i, doc in enumerate(documents):
                    # Relevance scoring
                    relevance = self.relevance_scorer(
                        question=question, document=doc.get("text", "")[:500]
                    )

                    # Quality assessment
                    quality = self.quality_assessor(document=doc.get("text", "")[:500])

                    # Extract scores (with fallbacks)
                    relevance_score = self._extract_score(
                        relevance.relevance_score, 0.5
                    )
                    quality_score = self._extract_score(quality.quality_score, 0.5)

                    scored_doc = {
                        **doc,
                        "relevance_score": relevance_score,
                        "quality_score": quality_score,
                        "relevance_reasoning": relevance.reasoning,
                        "quality_factors": quality.quality_factors,
                        "original_rank": i,
                    }
                    scored_documents.append(scored_doc)

                # Step 2: Calculate diversity scores
                diversity_result = self.diversity_calculator(
                    documents=str([doc["text"][:200] for doc in scored_documents[:20]])
                )
                diversity_scores = self._parse_diversity_scores(
                    diversity_result.diversity_score, len(scored_documents)
                )

                # Add diversity scores
                for i, doc in enumerate(scored_documents):
                    doc["diversity_score"] = (
                        diversity_scores[i] if i < len(diversity_scores) else 0.3
                    )

                # Step 3: Compute final composite scores
                for doc in scored_documents:
                    doc["composite_score"] = self._compute_composite_score(doc)

                # Step 4: Final ranking with LLM
                top_candidates = sorted(
                    scored_documents, key=lambda x: x["composite_score"], reverse=True
                )[: top_k * 2]

                final_ranking = self.final_ranker(
                    scores=str(
                        [
                            {
                                "id": doc.get("id", f"doc_{i}"),
                                "relevance": doc["relevance_score"],
                                "quality": doc["quality_score"],
                                "diversity": doc["diversity_score"],
                                "composite": doc["composite_score"],
                            }
                            for i, doc in enumerate(top_candidates[:10])
                        ]
                    ),
                    question=question,
                )

                # Sort by composite score and return top_k
                final_documents = sorted(
                    scored_documents, key=lambda x: x["composite_score"], reverse=True
                )[:top_k]

                return dspy.Prediction(
                    ranked_documents=final_documents,
                    ranking_scores={
                        "relevance_avg": sum(
                            d["relevance_score"] for d in final_documents
                        )
                        / len(final_documents),
                        "quality_avg": sum(d["quality_score"] for d in final_documents)
                        / len(final_documents),
                        "diversity_avg": sum(
                            d["diversity_score"] for d in final_documents
                        )
                        / len(final_documents),
                    },
                    ranking_reasoning=final_ranking.ranking_reasoning,
                )

            def _extract_score(self, score_text, default=0.5):
                """Extract numeric score from text."""
                try:
                    # Try to find a number in the text
                    numbers = re.findall(r"\d+\.?\d*", str(score_text))
                    if numbers:
                        score = float(numbers[0])
                        # Normalize to 0-1 range if needed
                        if score > 1:
                            score = score / 10 if score <= 10 else score / 100
                        return max(0, min(1, score))
                except Exception as e:
                    print(f"Error parsing score: {e}")
                    return default

            def _parse_diversity_scores(self, diversity_text, num_docs):
                """Parse diversity scores for documents."""
                try:
                    numbers = re.findall(r"\d+\.?\d*", str(diversity_text))
                    scores = [
                        float(n) / 10 if float(n) > 1 else float(n)
                        for n in numbers[:num_docs]
                    ]

                    # Fill remaining with default values
                    while len(scores) < num_docs:
                        scores.append(0.3)

                    return scores[:num_docs]
                except Exception as e:
                    print(f"Error parsing diversity scores: {e}")
                    return [0.3] * num_docs

            def _compute_composite_score(self, doc):
                """Compute weighted composite score."""
                weights = {"relevance": 0.5, "quality": 0.3, "diversity": 0.2}

                return (
                    weights["relevance"] * doc["relevance_score"]
                    + weights["quality"] * doc["quality_score"]
                    + weights["diversity"] * doc["diversity_score"]
                )

        class SemanticFilter(dspy.Module):
            """Semantic filtering for document relevance."""

            def __init__(self):
                super().__init__()
                self.semantic_matcher = dspy.ChainOfThought(
                    "question, document -> semantic_match, confidence"
                )
                self.topic_classifier = dspy.ChainOfThought(
                    "document -> topics, confidence"
                )

            def forward(self, question, documents, threshold=0.6):
                """Filter documents based on semantic relevance."""
                filtered_documents = []

                for doc in documents:
                    # Semantic matching
                    match_result = self.semantic_matcher(
                        question=question, document=doc.get("text", "")[:400]
                    )

                    # Topic classification
                    topic_result = self.topic_classifier(
                        document=doc.get("text", "")[:400]
                    )

                    # Extract confidence score
                    confidence = self._extract_confidence(match_result.confidence)

                    if confidence >= threshold:
                        filtered_doc = {
                            **doc,
                            "semantic_match": match_result.semantic_match,
                            "confidence": confidence,
                            "topics": topic_result.topics,
                            "filter_passed": True,
                        }
                        filtered_documents.append(filtered_doc)

                return dspy.Prediction(
                    filtered_documents=filtered_documents,
                    original_count=len(documents),
                    filtered_count=len(filtered_documents),
                    filter_threshold=threshold,
                )

            def _extract_confidence(self, confidence_text):
                """Extract confidence score from text."""
                try:
                    numbers = re.findall(r"\d+\.?\d*", str(confidence_text))
                    if numbers:
                        conf = float(numbers[0])
                        return conf / 100 if conf > 1 else conf
                except Exception as _:
                    pass
                return 0.5

        cell8_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Advanced Re-ranking System Created

                **Re-ranking Components:**  
                - **Relevance Scorer** - Assesses document relevance to query  
                - **Quality Assessor** - Evaluates document quality and credibility  
                - **Diversity Calculator** - Ensures result diversity and coverage  
                - **Final Ranker** - LLM-based final ranking optimization  

                **Scoring Criteria:**  
                - **Relevance (50%)** - How well document answers the question  
                - **Quality (30%)** - Document credibility, clarity, and completeness  
                - **Diversity (20%)** - Unique information contribution  

                **Semantic Filtering:**  
                - **Semantic Matching** - Deep semantic relevance assessment  
                - **Topic Classification** - Document topic identification  
                - **Confidence Thresholding** - Filters low-confidence matches  
                - **Adaptive Thresholds** - Adjustable filtering sensitivity  

                **Benefits:**  
                - **Higher Precision** - Better document relevance  
                - **Quality Control** - Filters low-quality content  
                - **Diverse Results** - Avoids redundant information  
                - **Explainable Ranking** - Clear scoring rationale  

                The advanced re-ranking system significantly improves retrieval quality!
                """
            )
        )
    else:
        cell8_desc = mo.md("")
        AdvancedReranker = None
        SemanticFilter = None
        cell8_content = mo.md("")

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return (AdvancedReranker,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        # Create individual UI components for testing interface
        query_input = mo.ui.text_area(
            placeholder="Enter a complex question to test advanced RAG patterns...",
            label="Test Query",
            rows=3,
        )

        pattern_selector = mo.ui.dropdown(
            options=[
                "Multi-Hop RAG",
                "Query Decomposition",
                "Hierarchical Retrieval",
                "Fusion RAG",
                "Adaptive RAG",
                "Advanced Re-ranking",
            ],
            label="RAG Pattern to Test",
            value="Multi-Hop RAG",
        )

        cell9_max_docs = mo.ui.slider(start=3, stop=15, value=8, label="Max Documents")

        test_button = mo.ui.run_button(label="🧪 Test Advanced RAG Pattern")

        # Create test interface layout
        cell9_test_layout = mo.vstack(
            [
                query_input,
                pattern_selector,
                cell9_max_docs,
                test_button,
            ]
        )

        cell9_desc = mo.md(
            cleandoc(
                """
                ## 🧪 Step 7: Advanced RAG Testing Interface

                Test the advanced RAG patterns with complex queries to see how they handle sophisticated information retrieval and reasoning tasks.
                """
            )
        )

        cell9_content = mo.vstack(
            [
                mo.md("### 🧪 Advanced RAG Pattern Testing"),
                mo.md("Test different advanced RAG patterns with complex queries:"),
                cell9_test_layout,
            ]
        )
    else:
        cell9_desc = mo.md("")
        query_input = None
        pattern_selector = None
        cell9_max_docs = None
        test_button = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack([cell9_desc, cell9_content])
    output.replace(cell9_out)
    return cell9_max_docs, pattern_selector, query_input, test_button


@app.cell
def _(
    AdaptiveRAG,
    AdvancedReranker,
    FusionRAG,
    HierarchicalRetriever,
    MultiHopRAG,
    QueryDecomposer,
    available_providers,
    cell9_max_docs,
    mo,
    output,
    pattern_selector,
    query_input,
    sample_hierarchy,
    test_button,
    time,
):
    if (
        available_providers
        and test_button
        and test_button.value
        and query_input
        and query_input.value.strip()
    ):
        try:
            # Get test parameters
            cell10_query = query_input.value.strip()
            cell10_pattern = pattern_selector.value
            cell10_max_docs = cell9_max_docs.value

            # Mock retriever for testing
            def mock_retriever(query, k=5):
                """Mock retriever that returns sample documents."""
                sample_docs = [
                    {
                        "id": "doc1",
                        "text": f"This document discusses {query} in detail with comprehensive analysis and examples.",
                        "title": "Analysis Document",
                    },
                    {
                        "id": "doc2",
                        "text": f"Research findings about {query} show interesting patterns and correlations.",
                        "title": "Research Paper",
                    },
                    {
                        "id": "doc3",
                        "text": f"Technical implementation of {query} requires careful consideration of various factors.",
                        "title": "Technical Guide",
                    },
                    {
                        "id": "doc4",
                        "text": f"Historical context of {query} provides important background information.",
                        "title": "Historical Context",
                    },
                    {
                        "id": "doc5",
                        "text": f"Future implications of {query} suggest significant developments ahead.",
                        "title": "Future Trends",
                    },
                ]
                return sample_docs[:k]

            # Test the selected pattern
            cell10_start_time = time.time()

            if cell10_pattern == "Multi-Hop RAG" and MultiHopRAG:
                rag_system = MultiHopRAG(mock_retriever)
                result = rag_system(cell10_query)
                cell10_result_text = f"**Answer:** {result.answer}\n\n**Hops:** {result.hops}\n\n**Sub-questions:** {', '.join(result.sub_questions)}"

            elif cell10_pattern == "Query Decomposition" and QueryDecomposer:
                decomposer = QueryDecomposer()
                result = decomposer(cell10_query)
                cell10_result_text = f"**Sub-queries:** {', '.join(result.sub_queries)}\n\n**Execution Plan:** {result.execution_plan}\n\n**Complexity:** {result.complexity_score}"

            elif cell10_pattern == "Hierarchical Retrieval" and HierarchicalRetriever:
                retriever = HierarchicalRetriever(sample_hierarchy)
                result = retriever(cell10_query, max_docs=cell10_max_docs)
                cell10_result_text = f"**Documents Found:** {len(result.documents)}\n\n**Domains Searched:** {', '.join(result.domains_searched)}\n\n**Categories:** {len(result.categories_searched)}"

            elif cell10_pattern == "Fusion RAG" and FusionRAG:
                fusion_retrievers = {
                    "semantic": mock_retriever,
                    "keyword": mock_retriever,
                    "hybrid": mock_retriever,
                }
                rag_system = FusionRAG(fusion_retrievers)
                result = rag_system(cell10_query)
                cell10_result_text = f"**Answer:** {result.answer}\n\n**Strategies Used:** {', '.join(result.strategies_used)}\n\n**Fusion Method:** {result.fusion_method}"

            elif cell10_pattern == "Adaptive RAG" and AdaptiveRAG:
                # Mock strategies for adaptive RAG
                mock_strategies = {
                    "basic": lambda q: type(
                        "Result", (), {"answer": f"Basic answer for: {q}"}
                    )(),
                    "multi_hop": lambda q: type(
                        "Result", (), {"answer": f"Multi-hop analysis of: {q}"}
                    )(),
                    "hierarchical": lambda q: type(
                        "Result", (), {"answer": f"Hierarchical breakdown of: {q}"}
                    )(),
                }
                rag_system = AdaptiveRAG(mock_strategies)
                result = rag_system(cell10_query)
                cell10_result_text = f"**Answer:** {result.answer}\n\n**Strategy Used:** {result.strategy_used}\n\n**Confidence:** {result.confidence}"

            elif cell10_pattern == "Advanced Re-ranking" and AdvancedReranker:
                reranker = AdvancedReranker()
                sample_docs = mock_retriever(cell10_query, k=cell10_max_docs)
                result = reranker(cell10_query, sample_docs, top_k=5)
                cell10_result_text = f"**Ranked Documents:** {len(result.ranked_documents)}\n\n**Avg Relevance:** {result.ranking_scores['relevance_avg']:.3f}\n\n**Avg Quality:** {result.ranking_scores['quality_avg']:.3f}"

            else:
                cell10_result_text = f"Pattern '{cell10_pattern}' is not available or not implemented yet."

            cell10_execution_time = time.time() - cell10_start_time

            cell10_out = mo.vstack(
                [
                    mo.md(f"### 🧪 Test Results: {cell10_pattern}"),
                    mo.md(f"**Query:** {cell10_query}"),
                    mo.md(cell10_result_text),
                    mo.md(f"**Execution Time:** {cell10_execution_time:.2f} seconds"),
                    mo.md("---"),
                    mo.md(
                        "*Try different patterns and queries to explore advanced RAG capabilities!*"
                    ),
                ]
            )

        except Exception as e:
            cell10_out = mo.vstack(
                [
                    mo.md("### ⚠️ Test Error"),
                    mo.md(f"An error occurred during testing: {str(e)}"),
                    mo.md("Please check your query and try again."),
                ]
            )
    else:
        cell10_out = mo.md(
            "*Enter a query above and click 'Test Advanced RAG Pattern' to see results*"
        )

    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell11_out = mo.md(
            cleandoc(
                """
                ## 🎉 Advanced RAG Patterns Complete!

                ### 🏆 What You've Built

                **Advanced RAG Systems:**  
                - ✅ **Multi-Hop RAG** - Chain multiple retrieval-generation cycles  
                - ✅ **Query Decomposition** - Break complex queries into sub-questions  
                - ✅ **Hierarchical Retrieval** - Multi-level document organization  
                - ✅ **Fusion RAG** - Combine multiple retrieval strategies  
                - ✅ **Adaptive RAG** - Dynamic strategy selection  
                - ✅ **Advanced Re-ranking** - Sophisticated document scoring  

                ### 🚀 Key Capabilities Mastered

                **Sophisticated Reasoning:**  
                - Multi-step question decomposition and planning  
                - Iterative retrieval with context accumulation  
                - Dynamic strategy selection based on query analysis  
                - Advanced document scoring and re-ranking  

                **System Architecture:**  
                - Modular, extensible RAG components  
                - Performance tracking and optimization  
                - Fallback mechanisms for robustness  
                - Explainable retrieval and ranking decisions  

                ### 🎯 Next Steps

                **Production Considerations:**  
                - **Scalability** - Optimize for large document collections  
                - **Caching** - Implement intelligent caching strategies  
                - **Monitoring** - Add comprehensive performance monitoring  
                - **Evaluation** - Develop robust evaluation frameworks  

                **Advanced Topics:**  
                - **Graph RAG** - Knowledge graph-based retrieval  
                - **Multimodal RAG** - Text, image, and audio integration  
                - **Real-time RAG** - Streaming and real-time processing  
                - **Federated RAG** - Distributed retrieval systems  

                ### 🌟 Congratulations!

                You've mastered advanced RAG patterns and built sophisticated information retrieval systems. These techniques enable handling complex, multi-faceted queries with intelligent strategy selection and high-quality results.

                 **Next Module:**
                ```bash
                uv run marimo run 03-retrieval-rag/rag_evaluation_optimization.py
                ```  
            
                **Ready to build production-grade RAG systems!** 🚀
                """
            )
        )
    else:
        cell11_out = mo.md("")

    output.replace(cell11_out)
    return


if __name__ == "__main__":
    app.run()
