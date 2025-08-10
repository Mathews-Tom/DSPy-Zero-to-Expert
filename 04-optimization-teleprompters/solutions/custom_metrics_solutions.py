# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo
"""Custom metrics design exercises solutions."""

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import re
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
            # 📊 Custom Metrics Design Solutions

            **Complete solutions for custom metrics design exercises.**

            ## 📚 Solutions Overview

            This notebook contains complete, working solutions for:  
            - Scientific paper evaluation metric with multi-dimensional scoring  
            - Code review quality metric with severity weighting  
            - Multi-language translation quality metric with cultural awareness  
            - Adaptive composite metric system with automatic weight adjustment  

            Study these solutions to master advanced evaluation techniques!
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
                ## ✅ Solutions Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  

                Ready to explore custom metrics solutions!
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
                ## 🎯 Solution 1: Scientific Paper Evaluation Metric

                **Complete implementation of comprehensive scientific paper summary evaluation.**
                """
            )
        )

        # Solution 1: Scientific Paper Evaluation Metric
        solution1_code = mo.ui.code_editor(
            value=cleandoc(
                """# Solution 1: Scientific Paper Evaluation Metric

                def scientific_paper_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate scientific paper summaries on multiple dimensions.

                    Evaluation criteria:
                    - Accuracy: Factual correctness of information (40% weight)
                    - Completeness: Coverage of key sections (30% weight)
                    - Clarity: Readability and coherence (20% weight)
                    - Technical precision: Proper use of scientific terminology (10% weight)
                    \"\"\"
                    try:
                        # Extract predicted and expected summaries
                        predicted_summary = getattr(pred, 'summary', '') or getattr(pred, 'output', '') or str(pred)
                        expected_summary = getattr(example, 'summary', '') or getattr(example, 'expected_output', '')

                        if not predicted_summary or not expected_summary:
                            return 0.0

                        # Convert to lowercase for comparison
                        pred_lower = predicted_summary.lower()
                        exp_lower = expected_summary.lower()

                        # 1. Accuracy scoring (40% weight)
                        # Use sequence matching to measure content overlap
                        matcher = difflib.SequenceMatcher(None, pred_lower, exp_lower)
                        content_similarity = matcher.ratio()

                        # Check for factual consistency (key terms present)
                        key_terms = extract_key_terms(expected_summary)
                        term_coverage = sum(1 for term in key_terms if term.lower() in pred_lower) / max(len(key_terms), 1)

                        accuracy_score = (content_similarity * 0.6 + term_coverage * 0.4) * 0.4

                        # 2. Completeness scoring (30% weight)
                        # Check for presence of key scientific sections
                        sections = {
                            'introduction': ['introduction', 'background', 'motivation', 'problem'],
                            'methodology': ['method', 'approach', 'technique', 'procedure', 'experiment'],
                            'results': ['result', 'finding', 'outcome', 'data', 'analysis'],
                            'conclusions': ['conclusion', 'summary', 'implication', 'future work']
                        }

                        section_scores = []
                        for section, keywords in sections.items():
                            section_present = any(keyword in pred_lower for keyword in keywords)
                            section_scores.append(1.0 if section_present else 0.0)

                        completeness_score = (sum(section_scores) / len(section_scores)) * 0.3

                        # 3. Clarity scoring (20% weight)
                        # Assess readability and structure
                        sentences = re.split(r'[.!?]+', predicted_summary)
                        sentences = [s.strip() for s in sentences if s.strip()]

                        # Sentence length variety (good writing has varied sentence lengths)
                        if sentences:
                            lengths = [len(s.split()) for s in sentences]
                            avg_length = sum(lengths) / len(lengths)
                            length_variety = len(set(lengths)) / len(lengths) if lengths else 0

                            # Optimal average sentence length for scientific writing: 15-25 words
                            length_score = 1.0 - abs(avg_length - 20) / 20 if avg_length <= 40 else 0.5
                            length_score = max(0, min(1, length_score))

                            clarity_score = (length_score * 0.6 + length_variety * 0.4) * 0.2
                        else:
                            clarity_score = 0.0

                        # 4. Technical precision scoring (10% weight)
                        # Check for proper scientific terminology and precision
                        technical_indicators = [
                            'significant', 'correlation', 'hypothesis', 'methodology', 'analysis',
                            'statistical', 'empirical', 'quantitative', 'qualitative', 'systematic'
                        ]

                        technical_count = sum(1 for indicator in technical_indicators if indicator in pred_lower)
                        technical_density = technical_count / max(len(predicted_summary.split()), 1)

                        # Optimal technical density: 2-5% of words should be technical terms
                        if 0.02 <= technical_density <= 0.05:
                            technical_score = 1.0 * 0.1
                        elif technical_density < 0.02:
                            technical_score = (technical_density / 0.02) * 0.1
                        else:
                            technical_score = max(0.5, 1.0 - (technical_density - 0.05) / 0.05) * 0.1

                        # Calculate weighted final score
                        final_score = accuracy_score + completeness_score + clarity_score + technical_score

                        return min(final_score, 1.0)  # Cap at 1.0

                    except Exception as e:
                        print(f"Error in scientific_paper_metric: {e}")
                        return 0.0

                def extract_key_terms(text):
                    \"\"\"Extract key scientific terms from text.\"\"\"
                    # Simple keyword extraction (in practice, use more sophisticated NLP)
                    words = re.findall(r'\\b[a-zA-Z]{4,}\\b', text.lower())

                    # Filter for likely scientific terms (longer words, specific patterns)
                    scientific_terms = []
                    for word in words:
                        if (len(word) >= 6 or 
                            word in ['data', 'test', 'study', 'model', 'theory', 'method'] or
                            word.endswith(('tion', 'ment', 'ness', 'ity', 'ism', 'ogy'))):
                            scientific_terms.append(word)

                    return list(set(scientific_terms))  # Remove duplicates

                # Test examples for scientific papers
                scientific_paper_examples = [
                    dspy.Example(
                        paper_title="Machine Learning in Climate Prediction",
                        abstract="This study investigates the application of deep learning models to improve climate prediction accuracy. We developed a novel neural network architecture that incorporates temporal and spatial features from satellite data.",
                        summary="The research presents a deep learning approach for climate prediction using satellite data. The methodology involves a novel neural network architecture that processes temporal and spatial features. Results show improved prediction accuracy compared to traditional methods. The study concludes that machine learning can significantly enhance climate forecasting capabilities.",
                        expected_output="This study develops a novel deep learning architecture for climate prediction using satellite data. The methodology combines temporal and spatial feature processing in neural networks. Results demonstrate superior accuracy compared to conventional approaches. The research concludes that machine learning significantly improves climate forecasting, with implications for better weather prediction and climate modeling."
                    ),
                    dspy.Example(
                        paper_title="Quantum Computing Applications in Cryptography",
                        abstract="We explore the potential impact of quantum computing on current cryptographic systems. Our analysis includes theoretical frameworks and practical implications for security protocols.",
                        summary="The paper examines quantum computing's effect on cryptography. It covers theoretical aspects and practical security implications. The study suggests that quantum computers could break current encryption methods.",
                        expected_output="This research analyzes quantum computing's transformative impact on cryptographic systems. The methodology encompasses theoretical framework analysis and practical security protocol evaluation. Results indicate that quantum computers pose significant threats to current encryption methods, requiring development of quantum-resistant algorithms. The study concludes with recommendations for transitioning to post-quantum cryptography to maintain security in the quantum era."
                    )
                ]

                def test_scientific_metric():
                    \"\"\"Test the scientific paper evaluation metric.\"\"\"
                    print("Testing Scientific Paper Evaluation Metric")
                    print("=" * 50)

                    for i, example in enumerate(scientific_paper_examples, 1):
                        # Create a prediction object
                        class MockPred:
                            def __init__(self, summary):
                                self.summary = summary
                                self.output = summary

                        pred = MockPred(example.summary)
                        score = scientific_paper_metric(example, pred)

                        print(f"\\nExample {i}: {example.paper_title}")
                        print(f"Score: {score:.3f}")
                        print(f"Summary length: {len(example.summary)} chars")
                        print(f"Expected length: {len(example.expected_output)} chars")

                        # Test with a poor summary
                        poor_pred = MockPred("This paper is about science and has some results.")
                        poor_score = scientific_paper_metric(example, poor_pred)
                        print(f"Poor summary score: {poor_score:.3f}")

                    print("\\nScientific paper metric test complete!")

                if __name__ == "__main__":
                    test_scientific_metric()
                """
            ),
            language="python",
            label="Solution 1 Code",
        )

        solution1_ui = mo.vstack([cell3_out, solution1_code])
    else:
        solution1_ui = mo.md("")

    output.replace(solution1_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell4_out = mo.md(
            cleandoc(
                """
                ## 🔧 Solution 2: Code Review Quality Metric

                **Complete implementation of code review comment evaluation system.**
                """
            )
        )

        # Solution 2: Code Review Quality Metric
        solution2_code = mo.ui.code_editor(
            value=cleandoc(
                """# Solution 2: Code Review Quality Metric

                def code_review_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate the quality of automated code review comments.

                    Evaluation criteria:
                    - Helpfulness: How useful is the comment for improving code (35% weight)
                    - Accuracy: Is the identified issue actually present (30% weight)
                    - Actionability: Can the developer act on the feedback (25% weight)
                    - Severity appropriateness: Is the severity level appropriate (10% weight)
                    \"\"\"
                    try:
                        # Extract review comment and expected feedback
                        predicted_comment = getattr(pred, 'comment', '') or getattr(pred, 'output', '') or str(pred)
                        expected_issues = getattr(example, 'expected_issues', [])
                        code_snippet = getattr(example, 'code', '')
                        issue_type = getattr(example, 'issue_type', 'general')

                        if not predicted_comment:
                            return 0.0

                        comment_lower = predicted_comment.lower()

                        # 1. Helpfulness scoring (35% weight)
                        helpfulness_indicators = {
                            'constructive': ['suggest', 'recommend', 'consider', 'try', 'could', 'might'],
                            'specific': ['line', 'function', 'variable', 'method', 'class', 'parameter'],
                            'educational': ['because', 'since', 'reason', 'why', 'explain', 'understand'],
                            'improvement': ['improve', 'better', 'optimize', 'enhance', 'refactor']
                        }

                        helpfulness_score = 0.0
                        for category, indicators in helpfulness_indicators.items():
                            category_score = min(sum(1 for indicator in indicators if indicator in comment_lower) / 2, 1.0)
                            helpfulness_score += category_score * 0.25  # Each category contributes equally

                        helpfulness_score = min(helpfulness_score, 1.0) * 0.35

                        # 2. Accuracy scoring (30% weight)
                        # Check if the comment identifies real issues present in the code
                        accuracy_score = 0.0

                        if expected_issues:
                            # Check if predicted comment mentions expected issues
                            issues_mentioned = 0
                            for issue in expected_issues:
                                issue_keywords = issue.lower().split()
                                if any(keyword in comment_lower for keyword in issue_keywords):
                                    issues_mentioned += 1

                            accuracy_score = (issues_mentioned / len(expected_issues)) * 0.3
                        else:
                            # If no specific issues provided, check for general code quality indicators
                            quality_indicators = ['bug', 'error', 'issue', 'problem', 'warning', 'style', 'performance']
                            if any(indicator in comment_lower for indicator in quality_indicators):
                                accuracy_score = 0.3

                        # 3. Actionability scoring (25% weight)
                        actionability_indicators = {
                            'specific_actions': ['change', 'add', 'remove', 'replace', 'move', 'rename'],
                            'clear_instructions': ['should', 'need to', 'must', 'required', 'necessary'],
                            'examples': ['example', 'like this', 'such as', 'for instance'],
                            'references': ['see', 'check', 'refer', 'documentation', 'standard']
                        }

                        actionability_score = 0.0
                        for category, indicators in actionability_indicators.items():
                            if any(indicator in comment_lower for indicator in indicators):
                                actionability_score += 0.25

                        actionability_score = min(actionability_score, 1.0) * 0.25

                        # 4. Severity appropriateness (10% weight)
                        severity_keywords = {
                            'critical': ['critical', 'severe', 'major', 'serious', 'urgent'],
                            'moderate': ['moderate', 'important', 'significant', 'notable'],
                            'minor': ['minor', 'small', 'trivial', 'cosmetic', 'style']
                        }

                        detected_severity = 'moderate'  # default
                        for severity, keywords in severity_keywords.items():
                            if any(keyword in comment_lower for keyword in keywords):
                                detected_severity = severity
                                break

                        # Map issue types to expected severity levels
                        expected_severity_map = {
                            'security': 'critical',
                            'bug': 'critical',
                            'performance': 'moderate',
                            'style': 'minor',
                            'documentation': 'minor',
                            'general': 'moderate'
                        }

                        expected_severity = expected_severity_map.get(issue_type, 'moderate')

                        # Score based on severity appropriateness
                        if detected_severity == expected_severity:
                            severity_score = 1.0 * 0.1
                        elif (detected_severity == 'critical' and expected_severity == 'moderate') or \\
                            (detected_severity == 'moderate' and expected_severity == 'minor'):
                            severity_score = 0.7 * 0.1  # Slightly over-estimated
                        elif (detected_severity == 'moderate' and expected_severity == 'critical') or \\
                            (detected_severity == 'minor' and expected_severity == 'moderate'):
                            severity_score = 0.5 * 0.1  # Under-estimated
                        else:
                            severity_score = 0.3 * 0.1  # Significantly off

                        # Calculate weighted final score
                        final_score = helpfulness_score + accuracy_score + actionability_score + severity_score

                        return min(final_score, 1.0)

                    except Exception as e:
                        print(f"Error in code_review_metric: {e}")
                        return 0.0

                # Test examples for code reviews
                code_review_examples = [
                    dspy.Example(
                        code=\"\"\"
                def calculate_total(items):
                    total = 0
                    for item in items:
                        total = total + item.price
                    return total
                        \"\"\",
                        expected_issues=["inefficient loop", "could use sum() function", "no input validation"],
                        issue_type="performance",
                        comment="This function could be optimized by using Python's built-in sum() function instead of manually iterating. Consider: return sum(item.price for item in items). Also add input validation to handle None or empty items list.",
                        expected_output="The function uses an inefficient manual loop that could be replaced with sum(). Recommend: return sum(item.price for item in items). Add input validation for None/empty items."
                    ),
                    dspy.Example(
                        code=\"\"\"
                def process_user_input(user_data):
                    sql = "SELECT * FROM users WHERE id = " + user_data['id']
                    return execute_query(sql)
                        \"\"\",
                        expected_issues=["SQL injection vulnerability", "security risk", "use parameterized queries"],
                        issue_type="security",
                        comment="Critical security vulnerability: SQL injection risk from string concatenation. Use parameterized queries instead: execute_query('SELECT * FROM users WHERE id = ?', [user_data['id']]). This prevents malicious SQL injection attacks.",
                        expected_output="Critical SQL injection vulnerability due to string concatenation. Must use parameterized queries to prevent security attacks."
                    ),
                    dspy.Example(
                        code=\"\"\"
                def getData():
                    return database.fetch_all()
                        \"\"\",
                        expected_issues=["poor naming convention", "missing docstring", "no error handling"],
                        issue_type="style",
                        comment="Function name should follow snake_case convention: get_data(). Add a docstring to explain the function's purpose. Consider adding error handling for database operations.",
                        expected_output="Use snake_case naming: get_data(). Add docstring and error handling for database operations."
                    )
                ]

                def test_code_review_metric():
                    \"\"\"Test the code review evaluation metric.\"\"\"
                    print("Testing Code Review Quality Metric")
                    print("=" * 50)

                    for i, example in enumerate(code_review_examples, 1):
                        # Create a prediction object
                        class MockPred:
                            def __init__(self, comment):
                                self.comment = comment
                                self.output = comment

                        pred = MockPred(example.comment)
                        score = code_review_metric(example, pred)

                        print(f"\\nExample {i}: {example.issue_type.title()} Issue")
                        print(f"Score: {score:.3f}")
                        print(f"Comment: {example.comment[:100]}...")
                        print(f"Expected issues: {', '.join(example.expected_issues)}")

                        # Test with a poor comment
                        poor_pred = MockPred("This code has problems.")
                        poor_score = code_review_metric(example, poor_pred)
                        print(f"Poor comment score: {poor_score:.3f}")

                    print("\\nCode review metric test complete!")

                if __name__ == "__main__":
                    test_code_review_metric()
                """
            ),
            language="python",
            label="Solution 2 Code",
        )

        solution2_ui = mo.vstack([cell4_out, solution2_code])
    else:
        solution2_ui = mo.md("")

    output.replace(solution2_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_out = mo.md(
            cleandoc(
                """
                ## ⚡ Solution 3: Multi-Language Translation Quality Metric

                **Complete implementation of sophisticated translation evaluation system.**
                """
            )
        )

        # Solution 3: Multi-Language Translation Quality Metric
        solution3_code = mo.ui.code_editor(
            value=cleandoc(
                """# Solution 3: Multi-Language Translation Quality Metric

                def translation_quality_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate translation quality across multiple languages.

                    Evaluation criteria:
                    - Fluency: Natural flow and grammar in target language (30% weight)
                    - Accuracy: Preservation of meaning from source (40% weight)
                    - Cultural appropriateness: Proper cultural context (20% weight)
                    - Terminology consistency: Consistent use of domain terms (10% weight)
                    \"\"\"
                    try:
                        # Extract translation details
                        source_text = getattr(example, 'source_text', '')
                        predicted_translation = getattr(pred, 'translation', '') or getattr(pred, 'output', '') or str(pred)
                        expected_translation = getattr(example, 'expected_translation', '')
                        target_language = getattr(example, 'target_language', 'unknown')
                        domain = getattr(example, 'domain', 'general')

                        if not predicted_translation or not expected_translation:
                            return 0.0

                        pred_lower = predicted_translation.lower()
                        exp_lower = expected_translation.lower()
                        source_lower = source_text.lower()

                        # 1. Fluency scoring (30% weight)
                        fluency_score = evaluate_fluency(predicted_translation, target_language)
                        fluency_score *= 0.3

                        # 2. Accuracy scoring (40% weight)
                        # Semantic similarity between source and translation
                        accuracy_score = evaluate_translation_accuracy(
                            source_text, predicted_translation, expected_translation
                        )
                        accuracy_score *= 0.4

                        # 3. Cultural appropriateness (20% weight)
                        cultural_score = evaluate_cultural_appropriateness(
                            predicted_translation, target_language, domain
                        )
                        cultural_score *= 0.2

                        # 4. Terminology consistency (10% weight)
                        terminology_score = evaluate_terminology_consistency(
                            source_text, predicted_translation, expected_translation, domain
                        )
                        terminology_score *= 0.1

                        # Apply language-specific adjustments
                        language_adjustment = get_language_adjustment_factor(target_language)

                        # Calculate weighted final score
                        final_score = (fluency_score + accuracy_score + cultural_score + terminology_score) * language_adjustment

                        return min(final_score, 1.0)

                    except Exception as e:
                        print(f"Error in translation_quality_metric: {e}")
                        return 0.0

                def evaluate_fluency(translation, target_language):
                    \"\"\"Evaluate the fluency of the translation.\"\"\"
                    try:
                        # Basic fluency indicators
                        sentences = re.split(r'[.!?]+', translation)
                        sentences = [s.strip() for s in sentences if s.strip()]

                        if not sentences:
                            return 0.0

                        # Check sentence structure
                        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

                        # Language-specific optimal sentence lengths
                        optimal_lengths = {
                            'english': (10, 25),
                            'spanish': (12, 28),
                            'french': (12, 30),
                            'german': (8, 22),  # German can have very long compound words
                            'chinese': (8, 20),
                            'japanese': (10, 25)
                        }

                        min_len, max_len = optimal_lengths.get(target_language.lower(), (10, 25))

                        # Score based on sentence length appropriateness
                        length_score = 0.0
                        for sentence in sentences:
                            sent_len = len(sentence.split())
                            if min_len <= sent_len <= max_len:
                                length_score += 1.0
                            elif sent_len < min_len:
                                length_score += sent_len / min_len
                            else:
                                length_score += max(0.5, 1.0 - (sent_len - max_len) / max_len)

                        length_score /= len(sentences)

                        # Check for grammatical indicators
                        grammar_indicators = {
                            'english': ['the', 'a', 'an', 'is', 'are', 'was', 'were'],
                            'spanish': ['el', 'la', 'los', 'las', 'es', 'son', 'está', 'están'],
                            'french': ['le', 'la', 'les', 'est', 'sont', 'était', 'étaient'],
                            'german': ['der', 'die', 'das', 'ist', 'sind', 'war', 'waren'],
                            'chinese': ['的', '是', '在', '有', '了', '和'],
                            'japanese': ['は', 'が', 'を', 'に', 'で', 'と']
                        }

                        indicators = grammar_indicators.get(target_language.lower(), [])
                        if indicators:
                            indicator_count = sum(1 for indicator in indicators if indicator in translation.lower())
                            grammar_score = min(indicator_count / 3, 1.0)  # Expect at least 3 indicators
                        else:
                            grammar_score = 0.7  # Default for unknown languages

                        return (length_score * 0.6 + grammar_score * 0.4)

                    except Exception:
                        return 0.5

                def evaluate_translation_accuracy(source_text, predicted_translation, expected_translation):
                    \"\"\"Evaluate how accurately the translation preserves meaning.\"\"\"
                    try:
                        # Compare with expected translation
                        matcher = difflib.SequenceMatcher(None, predicted_translation.lower(), expected_translation.lower())
                        similarity_score = matcher.ratio()

                        # Check for key concept preservation
                        # Extract important words (nouns, adjectives, verbs)
                        source_words = extract_important_words(source_text)
                        pred_words = extract_important_words(predicted_translation)
                        exp_words = extract_important_words(expected_translation)

                        # Measure concept overlap
                        if exp_words:
                            concept_overlap = len(set(pred_words) & set(exp_words)) / len(set(exp_words))
                        else:
                            concept_overlap = 0.5

                        # Check for meaning preservation (basic keyword matching)
                        meaning_score = 0.0
                        if source_words:
                            # This is simplified - in practice, use semantic similarity models
                            translated_concepts = len(set(pred_words)) / max(len(set(source_words)), 1)
                            meaning_score = min(translated_concepts, 1.0)

                        return (similarity_score * 0.4 + concept_overlap * 0.4 + meaning_score * 0.2)

                    except Exception:
                        return 0.5

                def evaluate_cultural_appropriateness(translation, target_language, domain):
                    \"\"\"Evaluate cultural appropriateness of the translation.\"\"\"
                    try:
                        translation_lower = translation.lower()

                        # Cultural indicators by language
                        cultural_markers = {
                            'spanish': {
                                'formal': ['usted', 'señor', 'señora', 'don', 'doña'],
                                'informal': ['tú', 'vos'],
                                'cultural_refs': ['familia', 'comunidad', 'tradición']
                            },
                            'japanese': {
                                'formal': ['です', 'ます', 'さん', '様'],
                                'informal': ['だ', 'である'],
                                'cultural_refs': ['和', '心', '道']
                            },
                            'french': {
                                'formal': ['vous', 'monsieur', 'madame'],
                                'informal': ['tu'],
                                'cultural_refs': ['culture', 'art', 'cuisine']
                            }
                        }

                        markers = cultural_markers.get(target_language.lower(), {})

                        if not markers:
                            return 0.7  # Default score for unknown languages

                        # Check for appropriate formality level
                        formal_count = sum(1 for marker in markers.get('formal', []) if marker in translation_lower)
                        informal_count = sum(1 for marker in markers.get('informal', []) if marker in translation_lower)

                        # Domain-specific appropriateness
                        domain_appropriateness = {
                            'business': 'formal',
                            'academic': 'formal',
                            'legal': 'formal',
                            'casual': 'informal',
                            'social': 'informal',
                            'general': 'mixed'
                        }

                        expected_style = domain_appropriateness.get(domain, 'mixed')

                        if expected_style == 'formal' and formal_count > informal_count:
                            formality_score = 1.0
                        elif expected_style == 'informal' and informal_count > formal_count:
                            formality_score = 1.0
                        elif expected_style == 'mixed':
                            formality_score = 0.8
                        else:
                            formality_score = 0.5

                        # Check for cultural references
                        cultural_refs = markers.get('cultural_refs', [])
                        cultural_ref_score = min(
                            sum(1 for ref in cultural_refs if ref in translation_lower) / max(len(cultural_refs), 1),
                            1.0
                        )

                        return (formality_score * 0.7 + cultural_ref_score * 0.3)

                    except Exception:
                        return 0.6

                def evaluate_terminology_consistency(source_text, predicted_translation, expected_translation, domain):
                    \"\"\"Evaluate consistency of domain-specific terminology.\"\"\"
                    try:
                        # Domain-specific terminology patterns
                        domain_terms = {
                            'medical': ['patient', 'diagnosis', 'treatment', 'symptom', 'therapy'],
                            'legal': ['contract', 'agreement', 'clause', 'liability', 'jurisdiction'],
                            'technical': ['system', 'process', 'function', 'parameter', 'interface'],
                            'business': ['revenue', 'profit', 'market', 'customer', 'strategy'],
                            'academic': ['research', 'analysis', 'methodology', 'conclusion', 'hypothesis']
                        }

                        relevant_terms = domain_terms.get(domain, [])

                        if not relevant_terms:
                            return 0.7  # Default for unknown domains

                        # Check if domain terms are consistently translated
                        source_lower = source_text.lower()
                        pred_lower = predicted_translation.lower()
                        exp_lower = expected_translation.lower()

                        # Count domain terms in source
                        source_term_count = sum(1 for term in relevant_terms if term in source_lower)

                        if source_term_count == 0:
                            return 0.8  # No domain terms to evaluate

                        # Check if predicted translation maintains domain terminology
                        # This is simplified - in practice, use bilingual dictionaries
                        pred_term_count = sum(1 for term in relevant_terms if term in pred_lower)
                        exp_term_count = sum(1 for term in relevant_terms if term in exp_lower)

                        # Score based on terminology preservation
                        if exp_term_count > 0:
                            consistency_score = min(pred_term_count / exp_term_count, 1.0)
                        else:
                            consistency_score = 0.5

                        return consistency_score

                    except Exception:
                        return 0.6

                def get_language_adjustment_factor(target_language):
                    \"\"\"Get language-specific adjustment factor.\"\"\"
                    # Some languages are inherently more difficult to translate
                    difficulty_adjustments = {
                        'chinese': 0.95,    # Character-based, different structure
                        'japanese': 0.95,   # Complex writing system, cultural context
                        'arabic': 0.95,     # Right-to-left, different grammar
                        'finnish': 0.97,    # Complex grammar, many cases
                        'hungarian': 0.97,  # Complex grammar
                        'english': 1.0,     # Baseline
                        'spanish': 1.0,     # Similar structure to English
                        'french': 1.0,      # Similar structure to English
                        'german': 0.98,     # Complex compound words
                        'italian': 1.0,     # Similar structure to English
                        'portuguese': 1.0   # Similar structure to English
                    }

                    return difficulty_adjustments.get(target_language.lower(), 0.98)

                def extract_important_words(text):
                    \"\"\"Extract important words (simplified - in practice use NLP).\"\"\"
                    # Remove common stop words and extract meaningful terms
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}

                    words = re.findall(r'\\b[a-zA-Z]{3,}\\b', text.lower())
                    important_words = [word for word in words if word not in stop_words and len(word) > 2]

                    return important_words

                # Test examples for translations
                translation_examples = [
                    dspy.Example(
                        source_text="The patient requires immediate medical attention for the diagnosis.",
                        target_language="spanish",
                        domain="medical",
                        translation="El paciente requiere atención médica inmediata para el diagnóstico.",
                        expected_translation="El paciente necesita atención médica inmediata para el diagnóstico."
                    ),
                    dspy.Example(
                        source_text="Please review the contract terms and conditions carefully.",
                        target_language="french",
                        domain="legal",
                        translation="Veuillez examiner attentivement les termes et conditions du contrat.",
                        expected_translation="Veuillez réviser soigneusement les termes et conditions du contrat."
                    ),
                    dspy.Example(
                        source_text="The system processes data efficiently using advanced algorithms.",
                        target_language="german",
                        domain="technical",
                        translation="Das System verarbeitet Daten effizient mit fortschrittlichen Algorithmen.",
                        expected_translation="Das System verarbeitet Daten effizient unter Verwendung fortgeschrittener Algorithmen."
                    )
                ]

                def test_translation_metric():
                    \"\"\"Test the translation quality evaluation metric.\"\"\"
                    print("Testing Multi-Language Translation Quality Metric")
                    print("=" * 60)

                    for i, example in enumerate(translation_examples, 1):
                        # Create a prediction object
                        class MockPred:
                            def __init__(self, translation):
                                self.translation = translation
                                self.output = translation

                        pred = MockPred(example.translation)
                        score = translation_quality_metric(example, pred)

                        print(f"\\nExample {i}: {example.source_text}")
                        print(f"Target Language: {example.target_language.title()}")
                        print(f"Domain: {example.domain.title()}")
                        print(f"Score: {score:.3f}")
                        print(f"Translation: {example.translation}")

                        # Test with a poor translation
                        poor_pred = MockPred("Bad translation here.")
                        poor_score = translation_quality_metric(example, poor_pred)
                        print(f"Poor translation score: {poor_score:.3f}")

                    print("\\nTranslation quality metric test complete!")

                if __name__ == "__main__":
                    test_translation_metric()
                """
            ),
            language="python",
            label="Solution 3 Code",
        )

        solution3_ui = mo.vstack([cell5_out, solution3_code])
    else:
        solution3_ui = mo.md("")

    output.replace(solution3_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell6_out = mo.md(
            cleandoc(
                """
                ## 📊 Solution 4: Adaptive Composite Metric System

                **Complete implementation of intelligent metric composition system.**
                """
            )
        )

        # Solution 4: Adaptive Composite Metric System
        solution4_code = mo.ui.code_editor(
            value=cleandoc(
                """# Solution 4: Adaptive Composite Metric System

                class CompositeMetricSystem:
                    \"\"\"
                    Adaptive composite metric system that adjusts weights based on task characteristics.
                    \"\"\"

                    def __init__(self):
                        self.base_metrics = {}
                        self.weight_profiles = {}
                        self.validation_results = []
                        self.performance_history = {}

                    def register_metric(self, name, metric_function, description=""):
                        \"\"\"Register a base metric function.\"\"\"
                        self.base_metrics[name] = {
                            'function': metric_function,
                            'description': description,
                            'performance_history': []
                        }
                        print(f"Registered metric: {name}")

                    def create_weight_profile(self, profile_name, task_characteristics, weights):
                        \"\"\"Create a weight profile for specific task characteristics.\"\"\"
                        # Validate weights sum to 1.0
                        total_weight = sum(weights.values())
                        if abs(total_weight - 1.0) > 0.01:
                            print(f"Warning: Weights sum to {total_weight:.3f}, normalizing...")
                            weights = {k: v/total_weight for k, v in weights.items()}

                        self.weight_profiles[profile_name] = {
                            'characteristics': task_characteristics,
                            'weights': weights,
                            'usage_count': 0,
                            'avg_performance': 0.0
                        }
                        print(f"Created weight profile: {profile_name}")

                    def auto_select_weights(self, task_type, content_characteristics):
                        \"\"\"Automatically select appropriate weights based on task characteristics.\"\"\"
                        best_profile = None
                        best_match_score = -1

                        for profile_name, profile in self.weight_profiles.items():
                            match_score = self._calculate_profile_match(
                                profile['characteristics'], 
                                {'task_type': task_type, **content_characteristics}
                            )

                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_profile = profile_name

                        if best_profile:
                            print(f"Selected weight profile: {best_profile} (match: {best_match_score:.2f})")
                            return self.weight_profiles[best_profile]['weights']
                        else:
                            # Return default equal weights
                            num_metrics = len(self.base_metrics)
                            default_weight = 1.0 / num_metrics if num_metrics > 0 else 1.0
                            return {name: default_weight for name in self.base_metrics.keys()}

                    def _calculate_profile_match(self, profile_chars, task_chars):
                        \"\"\"Calculate how well a profile matches task characteristics.\"\"\"
                        match_score = 0.0
                        total_chars = 0

                        for char_name, char_value in profile_chars.items():
                            if char_name in task_chars:
                                total_chars += 1
                                if task_chars[char_name] == char_value:
                                    match_score += 1.0
                                elif isinstance(char_value, str) and isinstance(task_chars[char_name], str):
                                    # Partial match for string similarity
                                    similarity = len(set(char_value.lower()) & set(task_chars[char_name].lower()))
                                    match_score += similarity / max(len(char_value), len(task_chars[char_name]))

                        return match_score / max(total_chars, 1)

                    def create_composite_metric(self, task_type, content_characteristics=None):
                        \"\"\"Create a composite metric optimized for the given task.\"\"\"
                        if content_characteristics is None:
                            content_characteristics = {}

                        # Get appropriate weights
                        weights = self.auto_select_weights(task_type, content_characteristics)

                        def composite_metric(example, pred, trace=None):
                            \"\"\"The actual composite metric function.\"\"\"
                            total_score = 0.0
                            total_weight = 0.0

                            for metric_name, weight in weights.items():
                                if metric_name in self.base_metrics:
                                    try:
                                        metric_func = self.base_metrics[metric_name]['function']
                                        score = metric_func(example, pred, trace)
                                        total_score += score * weight
                                        total_weight += weight
                                    except Exception as e:
                                        print(f"Error in metric {metric_name}: {e}")
                                        continue

                            return total_score / max(total_weight, 1.0)

                        # Store metadata about this composite metric
                        composite_metric.task_type = task_type
                        composite_metric.weights = weights
                        composite_metric.characteristics = content_characteristics

                        return composite_metric

                    def validate_metric(self, metric, test_examples, expected_scores=None):
                        \"\"\"Validate a metric against test examples.\"\"\"
                        validation_result = {
                            'metric_name': getattr(metric, '__name__', 'composite_metric'),
                            'test_count': len(test_examples),
                            'scores': [],
                            'avg_score': 0.0,
                            'score_variance': 0.0,
                            'correlation_with_expected': 0.0
                        }

                        # Run metric on test examples
                        scores = []
                        for example in test_examples:
                            try:
                                # Create a mock prediction for testing
                                pred = type('MockPred', (), {'output': getattr(example, 'expected_output', 'test output')})()
                                score = metric(example, pred)
                                scores.append(score)
                            except Exception as e:
                                print(f"Error validating example: {e}")
                                scores.append(0.0)

                        validation_result['scores'] = scores
                        validation_result['avg_score'] = sum(scores) / len(scores) if scores else 0.0

                        # Calculate variance
                        if len(scores) > 1:
                            mean_score = validation_result['avg_score']
                            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                            validation_result['score_variance'] = variance

                        # Calculate correlation with expected scores if provided
                        if expected_scores and len(expected_scores) == len(scores):
                            correlation = self._calculate_correlation(scores, expected_scores)
                            validation_result['correlation_with_expected'] = correlation

                        self.validation_results.append(validation_result)
                        return validation_result

                    def _calculate_correlation(self, scores1, scores2):
                        \"\"\"Calculate Pearson correlation coefficient.\"\"\"
                        if len(scores1) != len(scores2) or len(scores1) < 2:
                            return 0.0

                        mean1 = sum(scores1) / len(scores1)
                        mean2 = sum(scores2) / len(scores2)

                        numerator = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))

                        sum_sq1 = sum((s1 - mean1) ** 2 for s1 in scores1)
                        sum_sq2 = sum((s2 - mean2) ** 2 for s2 in scores2)

                        denominator = (sum_sq1 * sum_sq2) ** 0.5

                        return numerator / denominator if denominator > 0 else 0.0

                    def generate_metric_report(self, metric_name):
                        \"\"\"Generate a comprehensive report for a metric's performance.\"\"\"
                        if metric_name not in self.base_metrics:
                            return f"Metric '{metric_name}' not found."

                        metric_info = self.base_metrics[metric_name]

                        # Find validation results for this metric
                        relevant_validations = [
                            v for v in self.validation_results 
                            if v['metric_name'] == metric_name
                        ]

                        report = f\"\"\"
                METRIC PERFORMANCE REPORT: {metric_name}
                {'=' * 50}

                Description: {metric_info['description']}

                Validation Summary:
                - Total validations: {len(relevant_validations)}
                        \"\"\"

                        if relevant_validations:
                            avg_scores = [v['avg_score'] for v in relevant_validations]
                            overall_avg = sum(avg_scores) / len(avg_scores)

                            report += f\"\"\"
                - Overall average score: {overall_avg:.3f}
                - Best performance: {max(avg_scores):.3f}
                - Worst performance: {min(avg_scores):.3f}
                - Score consistency: {1.0 - (max(avg_scores) - min(avg_scores)):.3f}

                Recent Validation Results:
                            \"\"\"

                            for i, validation in enumerate(relevant_validations[-3:], 1):  # Last 3 validations
                                report += f\"\"\"
                Validation {i}:
                - Test examples: {validation['test_count']}
                - Average score: {validation['avg_score']:.3f}
                - Score variance: {validation['score_variance']:.3f}
                - Correlation with expected: {validation['correlation_with_expected']:.3f}
                                \"\"\"

                        report += \"\\n\\nRecommendations:\\n\"

                        if relevant_validations:
                            latest = relevant_validations[-1]
                            if latest['avg_score'] > 0.8:
                                report += "- Metric performs well, suitable for production use\\n"
                            elif latest['avg_score'] > 0.6:
                                report += "- Metric shows moderate performance, consider tuning\\n"
                            else:
                                report += "- Metric needs improvement, review implementation\\n"

                            if latest['score_variance'] > 0.1:
                                report += "- High score variance detected, check for edge cases\\n"

                            if latest['correlation_with_expected'] < 0.5:
                                report += "- Low correlation with expected scores, validate metric design\\n"

                        return report

                # Test examples for different task types
                adaptive_metric_examples = {
                    "qa": [
                        dspy.Example(
                            question="What is the capital of France?",
                            context="Geography question about European capitals",
                            answer="Paris",
                            expected_output="Paris is the capital of France."
                        ),
                        dspy.Example(
                            question="How does photosynthesis work?",
                            context="Biology question about plant processes",
                            answer="Plants convert sunlight to energy",
                            expected_output="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
                        )
                    ],
                    "summarization": [
                        dspy.Example(
                            text="Long article about climate change impacts on polar ice caps...",
                            summary="Climate change is causing polar ice caps to melt rapidly.",
                            expected_output="Climate change significantly impacts polar ice caps, causing accelerated melting and contributing to sea level rise."
                        )
                    ],
                    "classification": [
                        dspy.Example(
                            text="This movie was absolutely fantastic! Great acting and plot.",
                            category="positive",
                            expected_output="positive"
                        )
                    ],
                    "generation": [
                        dspy.Example(
                            prompt="Write a short story about a robot",
                            story="Once upon a time, there was a helpful robot named Alex...",
                            expected_output="A creative short story featuring a robot character with engaging narrative elements."
                        )
                    ]
                }

                def test_adaptive_metrics():
                    \"\"\"Test the adaptive composite metric system.\"\"\"
                    print("Testing Adaptive Composite Metric System")
                    print("=" * 50)

                    system = CompositeMetricSystem()

                    # Register base metrics
                    def accuracy_metric(example, pred, trace=None):
                        \"\"\"Simple accuracy metric.\"\"\"
                        expected = getattr(example, 'expected_output', '') or getattr(example, 'answer', '')
                        predicted = getattr(pred, 'output', '') or str(pred)
                        return 1.0 if expected.lower() in predicted.lower() else 0.0

                    def length_metric(example, pred, trace=None):
                        \"\"\"Output length appropriateness metric.\"\"\"
                        predicted = getattr(pred, 'output', '') or str(pred)
                        expected = getattr(example, 'expected_output', '') or getattr(example, 'answer', '')

                        pred_len = len(predicted.split())
                        exp_len = len(expected.split())

                        if exp_len == 0:
                            return 0.5

                        ratio = pred_len / exp_len
                        return max(0, 1.0 - abs(1.0 - ratio))

                    def relevance_metric(example, pred, trace=None):
                        \"\"\"Content relevance metric.\"\"\"
                        predicted = getattr(pred, 'output', '') or str(pred)
                        question = getattr(example, 'question', '') or getattr(example, 'prompt', '')

                        if not question:
                            return 0.5

                        # Simple keyword overlap
                        pred_words = set(predicted.lower().split())
                        question_words = set(question.lower().split())

                        overlap = len(pred_words & question_words)
                        return min(overlap / max(len(question_words), 1), 1.0)

                    # Register metrics
                    system.register_metric("accuracy", accuracy_metric, "Measures factual correctness")
                    system.register_metric("length", length_metric, "Evaluates output length appropriateness")
                    system.register_metric("relevance", relevance_metric, "Assesses content relevance")

                    # Create weight profiles for different tasks
                    system.create_weight_profile("qa_profile", 
                        {"task_type": "qa", "complexity": "medium"},
                        {"accuracy": 0.5, "length": 0.2, "relevance": 0.3}
                    )

                    system.create_weight_profile("summarization_profile",
                        {"task_type": "summarization", "complexity": "high"},
                        {"accuracy": 0.3, "length": 0.4, "relevance": 0.3}
                    )

                    system.create_weight_profile("generation_profile",
                        {"task_type": "generation", "complexity": "high"},
                        {"accuracy": 0.2, "length": 0.3, "relevance": 0.5}
                    )

                    # Test automatic weight selection and composite metric creation
                    print("\\nTesting automatic weight selection:")

                    for task_type in ["qa", "summarization", "generation"]:
                        print(f"\\n{task_type.upper()} Task:")

                        composite_metric = system.create_composite_metric(
                            task_type, 
                            {"complexity": "medium", "domain": "general"}
                        )

                        print(f"Selected weights: {composite_metric.weights}")

                        # Validate the metric
                        test_examples = adaptive_metric_examples.get(task_type, [])
                        if test_examples:
                            validation_result = system.validate_metric(composite_metric, test_examples)
                            print(f"Validation - Avg Score: {validation_result['avg_score']:.3f}, "
                                f"Variance: {validation_result['score_variance']:.3f}")

                    # Generate performance reports
                    print("\\n" + "="*30)
                    print("METRIC PERFORMANCE REPORTS")
                    print("="*30)

                    for metric_name in ["accuracy", "length", "relevance"]:
                        report = system.generate_metric_report(metric_name)
                        print(report)
                        print("-" * 30)

                    print("\\nAdaptive metric system test complete!")

                if __name__ == "__main__":
                    test_adaptive_metrics()
                """
            ),
            language="python",
            label="Solution 4 Code",
        )

        solution4_ui = mo.vstack([cell6_out, solution4_code])
    else:
        solution4_ui = mo.md("")

    output.replace(solution4_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_out = mo.md(
            cleandoc(
                """
                ## 🎓 Solutions Summary

                ### ✅ Complete Solutions Provided

                **Solution 1: Scientific Paper Evaluation Metric**  
                - ✅ Multi-dimensional evaluation (accuracy, completeness, clarity, technical precision)  
                - ✅ Domain-specific criteria for scientific content assessment  
                - ✅ Partial credit system with weighted scoring (40%, 30%, 20%, 10%)  
                - ✅ Robust error handling and key term extraction  

                **Solution 2: Code Review Quality Metric**  
                - ✅ Comprehensive evaluation (helpfulness, accuracy, actionability, severity)  
                - ✅ Severity weighting for different issue types (security, performance, style)  
                - ✅ False positive detection and constructive feedback assessment  
                - ✅ Domain-specific code quality indicators  

                **Solution 3: Multi-Language Translation Quality Metric**  
                - ✅ Advanced evaluation (fluency, accuracy, cultural appropriateness, terminology)  
                - ✅ Language-specific scoring adjustments and cultural awareness  
                - ✅ Domain-specific terminology consistency checking  
                - ✅ Sophisticated fluency assessment with grammar indicators  

                **Solution 4: Adaptive Composite Metric System**  
                - ✅ Intelligent metric composition with automatic weight adjustment  
                - ✅ Task characteristic-based profile matching system  
                - ✅ Comprehensive validation framework with correlation analysis  
                - ✅ Performance reporting and recommendation system  

                ### 🚀 Key Learning Points

                **Advanced Metric Design Principles:**  
                - **Multi-Dimensional Evaluation**: Always assess multiple aspects, not just accuracy  
                - **Domain Expertise Integration**: Incorporate domain-specific knowledge into metrics  
                - **Weighted Scoring**: Use appropriate weights based on task importance  
                - **Cultural and Contextual Awareness**: Consider cultural and contextual factors  

                **Implementation Best Practices:**  
                - **Robust Error Handling**: Always include comprehensive error handling  
                - **Partial Credit Systems**: Reward partially correct answers appropriately  
                - **Validation and Testing**: Thoroughly test metrics on diverse examples  
                - **Adaptive Systems**: Build systems that can adjust to different task types  

                **When to Use Custom Metrics:**  
                - **Domain-Specific Tasks**: When standard metrics don't capture domain nuances  
                - **Multi-Faceted Evaluation**: When you need to assess multiple quality dimensions  
                - **Cultural Sensitivity**: When cultural appropriateness matters  
                - **Complex Quality Assessment**: When simple accuracy isn't sufficient  

                **Metric Validation Strategies:**  
                - **Correlation Analysis**: Compare with human judgments or expected scores  
                - **Edge Case Testing**: Test on challenging and unusual examples  
                - **Cross-Domain Validation**: Test metrics across different domains  
                - **Performance Monitoring**: Track metric performance over time  

                These solutions provide a comprehensive foundation for designing
                sophisticated evaluation metrics that can significantly improve
                your DSPy optimization results! 🎉
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
