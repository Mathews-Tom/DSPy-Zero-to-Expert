#!/usr/bin/env python3
"""
Advanced Code Analysis and Generation Tool

This module provides comprehensive code analysis, quality assessment, and intelligent
code generation capabilities using DSPy. The system supports multiple programming
languages and provides advanced code understanding and improvement suggestions.

Learning Objectives:
- Implement advanced code parsing and analysis
- Create intelligent code quality assessment systems
- Build code generation and refactoring utilities
- Develop automated documentation and explanation generation
- Master DSPy patterns for code understanding and generation

Author: DSPy Learning Framework
"""

import ast
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


class AnalysisType(Enum):
    """Types of code analysis"""

    SYNTAX = "syntax"
    STYLE = "style"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEPENDENCIES = "dependencies"
    ARCHITECTURE = "architecture"


class IssueLevel(Enum):
    """Issue severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RefactoringType(Enum):
    """Types of refactoring operations"""

    EXTRACT_METHOD = "extract_method"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITION = "simplify_condition"
    REMOVE_DUPLICATION = "remove_duplication"
    OPTIMIZE_IMPORTS = "optimize_imports"
    ADD_TYPE_HINTS = "add_type_hints"
    IMPROVE_NAMING = "improve_naming"
    REDUCE_COMPLEXITY = "reduce_complexity"


@dataclass
class CodeIssue:
    """Represents a code issue or improvement suggestion"""

    id: str = field(default_factory=lambda: str(uuid4()))
    level: IssueLevel = IssueLevel.INFO
    category: str = ""
    message: str = ""
    description: str = ""
    file_path: str = ""
    line_number: int = 0
    column_number: int = 0
    code_snippet: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeMetrics:
    """Code quality metrics"""

    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    technical_debt_ratio: float = 0.0
    test_coverage: float = 0.0
    duplication_ratio: float = 0.0
    documentation_ratio: float = 0.0
    dependency_count: int = 0
    function_count: int = 0
    class_count: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RefactoringSuggestion:
    """Refactoring suggestion"""

    id: str = field(default_factory=lambda: str(uuid4()))
    refactoring_type: RefactoringType = RefactoringType.EXTRACT_METHOD
    title: str = ""
    description: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    original_code: str = ""
    refactored_code: str = ""
    benefits: List[str] = field(default_factory=list)
    confidence: float = 0.0
    estimated_effort: str = "low"  # low, medium, high
    impact_score: float = 0.0


@dataclass
class CodeAnalysisResult:
    """Complete code analysis result"""

    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    language: CodeLanguage = CodeLanguage.UNKNOWN
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    analysis_time: float = 0.0
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    issues: List[CodeIssue] = field(default_factory=list)
    suggestions: List[RefactoringSuggestion] = field(default_factory=list)
    quality_score: float = 0.0
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# DSPy Signatures for Code Analysis
class CodeQualityAnalysis(dspy.Signature):
    """Analyze code quality and provide improvement suggestions"""

    source_code: str = dspy.InputField(desc="Source code to analyze")
    language: str = dspy.InputField(desc="Programming language")
    analysis_focus: str = dspy.InputField(desc="Specific aspects to focus on")
    quality_score: float = dspy.OutputField(desc="Overall quality score (0-1)")
    issues: str = dspy.OutputField(desc="Identified issues and problems")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions")
    summary: str = dspy.OutputField(desc="Analysis summary")


class CodeExplanation(dspy.Signature):
    """Generate comprehensive code explanations"""

    source_code: str = dspy.InputField(desc="Source code to explain")
    language: str = dspy.InputField(desc="Programming language")
    explanation_level: str = dspy.InputField(desc="Explanation detail level")
    explanation: str = dspy.OutputField(desc="Detailed code explanation")
    key_concepts: str = dspy.OutputField(desc="Key programming concepts used")
    flow_description: str = dspy.OutputField(desc="Code execution flow")


class CodeGeneration(dspy.Signature):
    """Generate code based on specifications"""

    requirements: str = dspy.InputField(desc="Code requirements and specifications")
    language: str = dspy.InputField(desc="Target programming language")
    style_preferences: str = dspy.InputField(desc="Coding style preferences")
    generated_code: str = dspy.OutputField(desc="Generated source code")
    explanation: str = dspy.OutputField(desc="Code explanation and rationale")
    usage_examples: str = dspy.OutputField(desc="Usage examples and tests")


class CodeRefactoring(dspy.Signature):
    """Suggest and implement code refactoring"""

    original_code: str = dspy.InputField(desc="Original code to refactor")
    refactoring_goals: str = dspy.InputField(desc="Refactoring objectives")
    language: str = dspy.InputField(desc="Programming language")
    refactored_code: str = dspy.OutputField(desc="Refactored code")
    improvements: str = dspy.OutputField(desc="List of improvements made")
    rationale: str = dspy.OutputField(desc="Refactoring rationale")


class CodeDocumentation(dspy.Signature):
    """Generate comprehensive code documentation"""

    source_code: str = dspy.InputField(desc="Source code to document")
    language: str = dspy.InputField(desc="Programming language")
    doc_style: str = dspy.InputField(
        desc="Documentation style (docstring, comments, etc.)"
    )
    documentation: str = dspy.OutputField(desc="Generated documentation")
    api_reference: str = dspy.OutputField(desc="API reference if applicable")
    examples: str = dspy.OutputField(desc="Usage examples")


class CodeReview(dspy.Signature):
    """Perform comprehensive code review"""

    source_code: str = dspy.InputField(desc="Source code to review")
    language: str = dspy.InputField(desc="Programming language")
    review_criteria: str = dspy.InputField(desc="Review criteria and standards")
    review_summary: str = dspy.OutputField(desc="Overall review summary")
    strengths: str = dspy.OutputField(desc="Code strengths and good practices")
    weaknesses: str = dspy.OutputField(desc="Areas for improvement")
    recommendations: str = dspy.OutputField(desc="Specific recommendations")


class CodeLanguageDetector:
    """Detect programming language from code or file extension"""

    LANGUAGE_PATTERNS = {
        CodeLanguage.PYTHON: [
            r"def\s+\w+\s*\(",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r'if\s+__name__\s*==\s*["\']__main__["\']',
            r"class\s+\w+\s*\(",
        ],
        CodeLanguage.JAVASCRIPT: [
            r"function\s+\w+\s*\(",
            r"var\s+\w+\s*=",
            r"let\s+\w+\s*=",
            r"const\s+\w+\s*=",
            r"console\.log\s*\(",
        ],
        CodeLanguage.TYPESCRIPT: [
            r"interface\s+\w+\s*{",
            r"type\s+\w+\s*=",
            r":\s*\w+\s*=",
            r"export\s+interface",
            r'import\s+.*\s+from\s+["\']',
        ],
        CodeLanguage.JAVA: [
            r"public\s+class\s+\w+",
            r"public\s+static\s+void\s+main",
            r"import\s+java\.",
            r"System\.out\.println",
            r"@Override",
        ],
        CodeLanguage.CPP: [
            r"#include\s*<.*>",
            r"using\s+namespace\s+std",
            r"int\s+main\s*\(",
            r"std::\w+",
            r"cout\s*<<",
        ],
        CodeLanguage.C: [
            r"#include\s*<.*\.h>",
            r"int\s+main\s*\(",
            r"printf\s*\(",
            r"malloc\s*\(",
            r"struct\s+\w+\s*{",
        ],
        CodeLanguage.GO: [
            r"package\s+\w+",
            r"import\s+\(",
            r"func\s+\w+\s*\(",
            r"fmt\.Print",
            r"go\s+\w+\s*\(",
        ],
        CodeLanguage.RUST: [
            r"fn\s+\w+\s*\(",
            r"use\s+\w+::",
            r"let\s+mut\s+\w+",
            r"println!\s*\(",
            r"impl\s+\w+",
        ],
    }

    EXTENSION_MAP = {
        ".py": CodeLanguage.PYTHON,
        ".js": CodeLanguage.JAVASCRIPT,
        ".ts": CodeLanguage.TYPESCRIPT,
        ".java": CodeLanguage.JAVA,
        ".cpp": CodeLanguage.CPP,
        ".cc": CodeLanguage.CPP,
        ".cxx": CodeLanguage.CPP,
        ".c": CodeLanguage.C,
        ".go": CodeLanguage.GO,
        ".rs": CodeLanguage.RUST,
        ".rb": CodeLanguage.RUBY,
        ".php": CodeLanguage.PHP,
    }

    @classmethod
    def detect_from_file(cls, file_path: Union[str, Path]) -> CodeLanguage:
        """Detect language from file extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        return cls.EXTENSION_MAP.get(extension, CodeLanguage.UNKNOWN)

    @classmethod
    def detect_from_content(cls, code: str) -> CodeLanguage:
        """Detect language from code content"""
        scores = {}

        for language, patterns in cls.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
                score += matches
            scores[language] = score

        if not scores or max(scores.values()) == 0:
            return CodeLanguage.UNKNOWN

        return max(scores, key=scores.get)

    @classmethod
    def detect(
        cls, code: str, file_path: Optional[Union[str, Path]] = None
    ) -> CodeLanguage:
        """Detect language using both file extension and content"""
        if file_path:
            lang_from_file = cls.detect_from_file(file_path)
            if lang_from_file != CodeLanguage.UNKNOWN:
                return lang_from_file

        return cls.detect_from_content(code)


class CodeAnalyzer(ABC):
    """Abstract base class for language-specific code analyzers"""

    @abstractmethod
    def analyze_syntax(self, code: str) -> List[CodeIssue]:
        """Analyze code syntax"""
        pass

    @abstractmethod
    def calculate_metrics(self, code: str) -> CodeMetrics:
        """Calculate code metrics"""
        pass

    @abstractmethod
    def analyze_style(self, code: str) -> List[CodeIssue]:
        """Analyze code style"""
        pass

    @abstractmethod
    def get_supported_language(self) -> CodeLanguage:
        """Get supported language"""
        pass


class PythonCodeAnalyzer(CodeAnalyzer):
    """Python-specific code analyzer"""

    def get_supported_language(self) -> CodeLanguage:
        return CodeLanguage.PYTHON

    def analyze_syntax(self, code: str) -> List[CodeIssue]:
        """Analyze Python syntax"""
        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issue = CodeIssue(
                level=IssueLevel.ERROR,
                category="syntax",
                message=f"Syntax error: {e.msg}",
                description=str(e),
                line_number=e.lineno or 0,
                column_number=e.offset or 0,
                confidence=1.0,
            )
            issues.append(issue)

        return issues

    def calculate_metrics(self, code: str) -> CodeMetrics:
        """Calculate Python code metrics"""
        lines = code.split("\n")

        metrics = CodeMetrics()
        metrics.lines_of_code = len([line for line in lines if line.strip()])
        metrics.blank_lines = len([line for line in lines if not line.strip()])

        # Count comments
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped or "'''" in stripped:
                comment_lines += 1
        metrics.lines_of_comments = comment_lines

        # Calculate documentation ratio
        if metrics.lines_of_code > 0:
            metrics.documentation_ratio = comment_lines / metrics.lines_of_code

        # Parse AST for more detailed metrics
        try:
            tree = ast.parse(code)

            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics.function_count += 1
                elif isinstance(node, ast.ClassDef):
                    metrics.class_count += 1

            # Simple complexity calculation
            complexity = self._calculate_complexity(tree)
            metrics.cyclomatic_complexity = complexity

            # Simple maintainability index (simplified formula)
            if metrics.lines_of_code > 0:
                metrics.maintainability_index = (
                    max(
                        0,
                        171
                        - 5.2 * complexity
                        - 0.23 * metrics.lines_of_code
                        - 16.2 * metrics.lines_of_comments,
                    )
                    / 171
                )

        except SyntaxError:
            pass

        return metrics

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def analyze_style(self, code: str) -> List[CodeIssue]:
        """Analyze Python code style"""
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # PEP 8 recommends 79, but 88 is common
                issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        category="style",
                        message="Line too long",
                        description=f"Line exceeds 88 characters ({len(line)} chars)",
                        line_number=i,
                        code_snippet=line,
                        confidence=0.9,
                    )
                )

            # Check for trailing whitespace
            if line.endswith(" ") or line.endswith("\t"):
                issues.append(
                    CodeIssue(
                        level=IssueLevel.INFO,
                        category="style",
                        message="Trailing whitespace",
                        description="Line has trailing whitespace",
                        line_number=i,
                        code_snippet=line,
                        confidence=1.0,
                    )
                )

            # Check for multiple statements on one line
            if ";" in line and not line.strip().startswith("#"):
                issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        category="style",
                        message="Multiple statements on one line",
                        description="Avoid multiple statements on a single line",
                        line_number=i,
                        code_snippet=line,
                        confidence=0.8,
                    )
                )

        return issues


class JavaScriptCodeAnalyzer(CodeAnalyzer):
    """JavaScript-specific code analyzer"""

    def get_supported_language(self) -> CodeLanguage:
        return CodeLanguage.JAVASCRIPT

    def analyze_syntax(self, code: str) -> List[CodeIssue]:
        """Analyze JavaScript syntax (simplified)"""
        issues = []

        # Basic syntax checks
        if code.count("{") != code.count("}"):
            issues.append(
                CodeIssue(
                    level=IssueLevel.ERROR,
                    category="syntax",
                    message="Mismatched braces",
                    description="Number of opening and closing braces don't match",
                    confidence=0.9,
                )
            )

        if code.count("(") != code.count(")"):
            issues.append(
                CodeIssue(
                    level=IssueLevel.ERROR,
                    category="syntax",
                    message="Mismatched parentheses",
                    description="Number of opening and closing parentheses don't match",
                    confidence=0.9,
                )
            )

        return issues

    def calculate_metrics(self, code: str) -> CodeMetrics:
        """Calculate JavaScript code metrics"""
        lines = code.split("\n")

        metrics = CodeMetrics()
        metrics.lines_of_code = len([line for line in lines if line.strip()])
        metrics.blank_lines = len([line for line in lines if not line.strip()])

        # Count comments
        comment_lines = 0
        in_block_comment = False

        for line in lines:
            stripped = line.strip()
            if "/*" in stripped:
                in_block_comment = True
            if "*/" in stripped:
                in_block_comment = False
                comment_lines += 1
                continue

            if in_block_comment or stripped.startswith("//"):
                comment_lines += 1

        metrics.lines_of_comments = comment_lines

        # Count functions
        function_patterns = [
            r"function\s+\w+\s*\(",
            r"\w+\s*:\s*function\s*\(",
            r"\w+\s*=\s*function\s*\(",
            r"\w+\s*=>\s*{",
        ]

        for pattern in function_patterns:
            matches = re.findall(pattern, code)
            metrics.function_count += len(matches)

        # Simple complexity calculation
        complexity_indicators = [
            "{",
            "if",
            "else",
            "for",
            "while",
            "switch",
            "case",
            "&&",
            "||",
        ]
        complexity = 1
        for indicator in complexity_indicators:
            complexity += code.count(indicator)

        metrics.cyclomatic_complexity = complexity / 10  # Normalize

        return metrics

    def analyze_style(self, code: str) -> List[CodeIssue]:
        """Analyze JavaScript code style"""
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for var usage (prefer let/const)
            if re.search(r"\bvar\s+\w+", line):
                issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        category="style",
                        message="Use 'let' or 'const' instead of 'var'",
                        description="Modern JavaScript prefers let/const over var",
                        line_number=i,
                        code_snippet=line,
                        confidence=0.8,
                    )
                )

            # Check for == instead of ===
            if (
                "==" in line
                and "===" not in line
                and "!=" in line
                and "!==" not in line
            ):
                issues.append(
                    CodeIssue(
                        level=IssueLevel.WARNING,
                        category="style",
                        message="Use strict equality (===) instead of ==",
                        description="Strict equality avoids type coercion issues",
                        line_number=i,
                        code_snippet=line,
                        confidence=0.9,
                    )
                )

        return issues


class CodeAnalyzerRegistry:
    """Registry for code analyzers"""

    def __init__(self):
        self.analyzers: Dict[CodeLanguage, CodeAnalyzer] = {}
        self._register_default_analyzers()

    def _register_default_analyzers(self):
        """Register default analyzers"""
        analyzers = [
            PythonCodeAnalyzer(),
            JavaScriptCodeAnalyzer(),
        ]

        for analyzer in analyzers:
            self.analyzers[analyzer.get_supported_language()] = analyzer

    def register_analyzer(self, analyzer: CodeAnalyzer):
        """Register a new analyzer"""
        language = analyzer.get_supported_language()
        self.analyzers[language] = analyzer
        logger.info(f"Registered analyzer for {language.value}")

    def get_analyzer(self, language: CodeLanguage) -> Optional[CodeAnalyzer]:
        """Get analyzer for language"""
        return self.analyzers.get(language)

    def get_supported_languages(self) -> List[CodeLanguage]:
        """Get list of supported languages"""
        return list(self.analyzers.keys())


class CodeAnalysisEngine:
    """Main code analysis engine"""

    def __init__(self):
        self.analyzer_registry = CodeAnalyzerRegistry()
        self.language_detector = CodeLanguageDetector()
        self.analysis_results: Dict[str, CodeAnalysisResult] = {}

        # Initialize DSPy modules
        self.quality_analyzer = dspy.ChainOfThought(CodeQualityAnalysis)
        self.code_explainer = dspy.ChainOfThought(CodeExplanation)
        self.code_generator = dspy.ChainOfThought(CodeGeneration)
        self.code_refactorer = dspy.ChainOfThought(CodeRefactoring)
        self.code_documenter = dspy.ChainOfThought(CodeDocumentation)
        self.code_reviewer = dspy.ChainOfThought(CodeReview)

    async def analyze_code(
        self,
        code: str,
        file_path: Optional[str] = None,
        analysis_types: List[AnalysisType] = None,
    ) -> CodeAnalysisResult:
        """Perform comprehensive code analysis"""
        start_time = asyncio.get_event_loop().time()

        # Detect language
        language = self.language_detector.detect(code, file_path)

        # Initialize result
        result = CodeAnalysisResult(
            file_path=file_path or "", language=language, analyzed_at=datetime.utcnow()
        )

        try:
            # Get appropriate analyzer
            analyzer = self.analyzer_registry.get_analyzer(language)

            if analyzer:
                # Perform static analysis
                result.issues.extend(analyzer.analyze_syntax(code))
                result.issues.extend(analyzer.analyze_style(code))
                result.metrics = analyzer.calculate_metrics(code)
            else:
                result.errors.append(f"No analyzer available for {language.value}")

            # Set default analysis types
            if analysis_types is None:
                analysis_types = [
                    AnalysisType.SYNTAX,
                    AnalysisType.STYLE,
                    AnalysisType.COMPLEXITY,
                    AnalysisType.MAINTAINABILITY,
                ]

            # Perform AI-powered analysis
            await self._perform_ai_analysis(result, code, analysis_types)

            # Calculate overall quality score
            result.quality_score = self._calculate_quality_score(result)

            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Code analysis failed: {e}")

        finally:
            result.analysis_time = asyncio.get_event_loop().time() - start_time
            self.analysis_results[result.analysis_id] = result

        return result

    async def _perform_ai_analysis(
        self, result: CodeAnalysisResult, code: str, analysis_types: List[AnalysisType]
    ):
        """Perform AI-powered code analysis"""
        try:
            # Truncate code for AI analysis
            max_code_length = 4000
            analysis_code = (
                code[:max_code_length] if len(code) > max_code_length else code
            )

            # Quality analysis
            quality_result = self.quality_analyzer(
                source_code=analysis_code,
                language=result.language.value,
                analysis_focus="quality, maintainability, best practices",
            )

            result.summary = quality_result.summary

            # Parse AI-generated issues
            issues_text = quality_result.issues
            ai_issues = self._parse_ai_issues(issues_text, result.language)
            result.issues.extend(ai_issues)

            # Parse AI-generated suggestions
            suggestions_text = quality_result.suggestions
            ai_suggestions = self._parse_ai_suggestions(
                suggestions_text, result.language
            )
            result.suggestions.extend(ai_suggestions)

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")

    def _parse_ai_issues(
        self, issues_text: str, language: CodeLanguage
    ) -> List[CodeIssue]:
        """Parse AI-generated issues"""
        issues = []

        # Simple parsing - in production, use more sophisticated parsing
        lines = issues_text.split("\n")

        for line in lines:
            if line.strip() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    category = parts[0].strip().lower()
                    message = parts[1].strip()

                    # Determine issue level
                    level = IssueLevel.INFO
                    if any(
                        word in message.lower()
                        for word in ["error", "critical", "severe"]
                    ):
                        level = IssueLevel.ERROR
                    elif any(
                        word in message.lower()
                        for word in ["warning", "caution", "potential"]
                    ):
                        level = IssueLevel.WARNING

                    issue = CodeIssue(
                        level=level,
                        category=category,
                        message=message,
                        confidence=0.7,  # AI-generated confidence
                    )
                    issues.append(issue)

        return issues

    def _parse_ai_suggestions(
        self, suggestions_text: str, language: CodeLanguage
    ) -> List[RefactoringSuggestion]:
        """Parse AI-generated refactoring suggestions"""
        suggestions = []

        # Simple parsing
        lines = suggestions_text.split("\n")

        for line in lines:
            if line.strip() and (
                "refactor" in line.lower() or "improve" in line.lower()
            ):
                suggestion = RefactoringSuggestion(
                    title=line.strip(),
                    description=line.strip(),
                    confidence=0.7,
                    estimated_effort="medium",
                )
                suggestions.append(suggestion)

        return suggestions

    def _calculate_quality_score(self, result: CodeAnalysisResult) -> float:
        """Calculate overall quality score"""
        score = 1.0  # Start with perfect score

        # Deduct for issues
        for issue in result.issues:
            if issue.level == IssueLevel.CRITICAL:
                score -= 0.2
            elif issue.level == IssueLevel.ERROR:
                score -= 0.1
            elif issue.level == IssueLevel.WARNING:
                score -= 0.05
            elif issue.level == IssueLevel.INFO:
                score -= 0.01

        # Consider metrics
        if result.metrics.cyclomatic_complexity > 10:
            score -= 0.1

        if result.metrics.documentation_ratio < 0.1:
            score -= 0.1

        return max(0.0, score)

    def _generate_recommendations(self, result: CodeAnalysisResult) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Based on issues
        error_count = len([i for i in result.issues if i.level == IssueLevel.ERROR])
        warning_count = len([i for i in result.issues if i.level == IssueLevel.WARNING])

        if error_count > 0:
            recommendations.append(
                f"Fix {error_count} critical errors before proceeding"
            )

        if warning_count > 5:
            recommendations.append(
                f"Address {warning_count} warnings to improve code quality"
            )

        # Based on metrics
        if result.metrics.cyclomatic_complexity > 10:
            recommendations.append(
                "Reduce code complexity by breaking down large functions"
            )

        if result.metrics.documentation_ratio < 0.1:
            recommendations.append("Add more comments and documentation")

        if result.metrics.lines_of_code > 1000:
            recommendations.append(
                "Consider splitting large files into smaller modules"
            )

        return recommendations

    async def explain_code(
        self,
        code: str,
        language: Optional[CodeLanguage] = None,
        explanation_level: str = "detailed",
    ) -> Dict[str, Any]:
        """Generate comprehensive code explanation"""
        if language is None:
            language = self.language_detector.detect(code)

        try:
            explanation_result = self.code_explainer(
                source_code=code[:3000],  # Truncate for processing
                language=language.value,
                explanation_level=explanation_level,
            )

            return {
                "language": language.value,
                "explanation": explanation_result.explanation,
                "key_concepts": explanation_result.key_concepts,
                "flow_description": explanation_result.flow_description,
                "explanation_level": explanation_level,
            }

        except Exception as e:
            return {"error": str(e), "language": language.value}

    async def generate_code(
        self,
        requirements: str,
        language: CodeLanguage,
        style_preferences: str = "clean, readable, well-documented",
    ) -> Dict[str, Any]:
        """Generate code based on requirements"""
        try:
            generation_result = self.code_generator(
                requirements=requirements,
                language=language.value,
                style_preferences=style_preferences,
            )

            return {
                "language": language.value,
                "generated_code": generation_result.generated_code,
                "explanation": generation_result.explanation,
                "usage_examples": generation_result.usage_examples,
                "requirements": requirements,
            }

        except Exception as e:
            return {
                "error": str(e),
                "language": language.value,
                "requirements": requirements,
            }

    async def refactor_code(
        self, code: str, refactoring_goals: str, language: Optional[CodeLanguage] = None
    ) -> Dict[str, Any]:
        """Refactor code based on goals"""
        if language is None:
            language = self.language_detector.detect(code)

        try:
            refactoring_result = self.code_refactorer(
                original_code=code[:3000],  # Truncate for processing
                refactoring_goals=refactoring_goals,
                language=language.value,
            )

            return {
                "language": language.value,
                "original_code": code,
                "refactored_code": refactoring_result.refactored_code,
                "improvements": refactoring_result.improvements,
                "rationale": refactoring_result.rationale,
                "refactoring_goals": refactoring_goals,
            }

        except Exception as e:
            return {
                "error": str(e),
                "language": language.value,
                "refactoring_goals": refactoring_goals,
            }

    async def generate_documentation(
        self,
        code: str,
        language: Optional[CodeLanguage] = None,
        doc_style: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Generate comprehensive documentation for code"""
        if language is None:
            language = self.language_detector.detect(code)

        try:
            doc_result = self.code_documenter(
                source_code=code[:3000],  # Truncate for processing
                language=language.value,
                doc_style=doc_style,
            )

            return {
                "language": language.value,
                "documentation": doc_result.documentation,
                "api_reference": doc_result.api_reference,
                "examples": doc_result.examples,
                "doc_style": doc_style,
            }

        except Exception as e:
            return {"error": str(e), "language": language.value, "doc_style": doc_style}

    async def review_code(
        self,
        code: str,
        language: Optional[CodeLanguage] = None,
        review_criteria: str = "best practices, security, performance",
    ) -> Dict[str, Any]:
        """Perform comprehensive code review"""
        if language is None:
            language = self.language_detector.detect(code)

        try:
            review_result = self.code_reviewer(
                source_code=code[:3000],  # Truncate for processing
                language=language.value,
                review_criteria=review_criteria,
            )

            return {
                "language": language.value,
                "review_summary": review_result.review_summary,
                "strengths": review_result.strengths,
                "weaknesses": review_result.weaknesses,
                "recommendations": review_result.recommendations,
                "review_criteria": review_criteria,
            }

        except Exception as e:
            return {
                "error": str(e),
                "language": language.value,
                "review_criteria": review_criteria,
            }

    def get_analysis_result(self, analysis_id: str) -> Optional[CodeAnalysisResult]:
        """Get analysis result by ID"""
        return self.analysis_results.get(analysis_id)

    def list_analyses(self) -> List[Dict[str, Any]]:
        """List all performed analyses"""
        analyses = []
        for analysis_id, result in self.analysis_results.items():
            analyses.append(
                {
                    "analysis_id": analysis_id,
                    "file_path": result.file_path,
                    "language": result.language.value,
                    "analyzed_at": result.analyzed_at.isoformat(),
                    "quality_score": result.quality_score,
                    "issue_count": len(result.issues),
                    "suggestion_count": len(result.suggestions),
                }
            )
        return analyses


class CodeAnalysisAPI:
    """REST API interface for code analysis system"""

    def __init__(self):
        self.engine = CodeAnalysisEngine()

    async def analyze_code_endpoint(
        self,
        code: str,
        file_path: Optional[str] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """API endpoint for code analysis"""
        try:
            if analysis_types:
                analysis_types = [AnalysisType(at) for at in analysis_types]

            result = await self.engine.analyze_code(code, file_path, analysis_types)

            return {
                "success": True,
                "analysis_id": result.analysis_id,
                "language": result.language.value,
                "quality_score": result.quality_score,
                "analysis_time": result.analysis_time,
                "metrics": {
                    "lines_of_code": result.metrics.lines_of_code,
                    "cyclomatic_complexity": result.metrics.cyclomatic_complexity,
                    "maintainability_index": result.metrics.maintainability_index,
                    "documentation_ratio": result.metrics.documentation_ratio,
                    "function_count": result.metrics.function_count,
                    "class_count": result.metrics.class_count,
                },
                "issue_summary": {
                    "total": len(result.issues),
                    "critical": len(
                        [i for i in result.issues if i.level == IssueLevel.CRITICAL]
                    ),
                    "error": len(
                        [i for i in result.issues if i.level == IssueLevel.ERROR]
                    ),
                    "warning": len(
                        [i for i in result.issues if i.level == IssueLevel.WARNING]
                    ),
                    "info": len(
                        [i for i in result.issues if i.level == IssueLevel.INFO]
                    ),
                },
                "summary": result.summary,
                "recommendations": result.recommendations[:5],  # Limit for API
                "suggestion_count": len(result.suggestions),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def explain_code_endpoint(
        self,
        code: str,
        language: Optional[str] = None,
        explanation_level: str = "detailed",
    ) -> Dict[str, Any]:
        """API endpoint for code explanation"""
        try:
            lang = CodeLanguage(language) if language else None
            result = await self.engine.explain_code(code, lang, explanation_level)

            return {"success": True, **result}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def generate_code_endpoint(
        self,
        requirements: str,
        language: str,
        style_preferences: str = "clean, readable",
    ) -> Dict[str, Any]:
        """API endpoint for code generation"""
        try:
            lang = CodeLanguage(language)
            result = await self.engine.generate_code(
                requirements, lang, style_preferences
            )

            return {"success": True, **result}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}


# Demo and Testing Functions
async def demo_code_analysis():
    """Demonstrate code analysis capabilities"""
    print("🔍 Code Analysis and Generation Tool Demo")
    print("=" * 50)

    # Initialize engine
    engine = CodeAnalysisEngine()

    # Sample code examples
    sample_codes = {
        "python_example.py": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    # Calculate first 10 fibonacci numbers
    for i in range(10):
        result = calculate_fibonacci(i)
        print(f"Fibonacci({i}) = {result}")

if __name__ == "__main__":
    main()
        """,
        "javascript_example.js": """
function calculateFactorial(n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    var result = 1;
    for (var i = 2; i <= n; i++) {
        result = result * i;
    }
    return result;
}

console.log("Factorial of 5:", calculateFactorial(5));
        """,
        "complex_python.py": """
import os
import sys

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed_data = []
    
    def process(self):
        for item in self.data:
            if item > 0:
                if item % 2 == 0:
                    if item > 10:
                        self.processed_data.append(item * 2)
                    else:
                        self.processed_data.append(item)
                else:
                    if item > 5:
                        self.processed_data.append(item + 1)
                    else:
                        self.processed_data.append(item - 1)
        return self.processed_data

def main():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    processor = DataProcessor(data)
    result = processor.process()
    print(result)

main()
        """,
    }

    print("📊 Analyzing sample code files...")

    # Analyze each code sample
    for filename, code in sample_codes.items():
        print(f"\n🔄 Analyzing: {filename}")

        result = await engine.analyze_code(code, filename)

        print(f"   Language: {result.language.value}")
        print(f"   Quality Score: {result.quality_score:.2f}")
        print(f"   Lines of Code: {result.metrics.lines_of_code}")
        print(f"   Complexity: {result.metrics.cyclomatic_complexity:.1f}")
        print(f"   Functions: {result.metrics.function_count}")
        print(f"   Classes: {result.metrics.class_count}")
        print(f"   Issues Found: {len(result.issues)}")

        # Show top issues
        if result.issues:
            print("   Top Issues:")
            for issue in result.issues[:3]:
                print(f"     - {issue.level.value.upper()}: {issue.message}")

        # Show recommendations
        if result.recommendations:
            print("   Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"     - {rec}")

    # Demonstrate code explanation
    print(f"\n📖 Code Explanation Demo")
    explanation = await engine.explain_code(
        sample_codes["python_example.py"], explanation_level="beginner"
    )

    if "error" not in explanation:
        print(f"   Explanation: {explanation['explanation'][:200]}...")
        print(f"   Key Concepts: {explanation['key_concepts'][:100]}...")

    # Demonstrate code generation
    print(f"\n🔧 Code Generation Demo")
    generation = await engine.generate_code(
        "Create a Python function that sorts a list of numbers using bubble sort algorithm",
        CodeLanguage.PYTHON,
    )

    if "error" not in generation:
        print(f"   Generated Code Preview:")
        print("   " + "\n   ".join(generation["generated_code"][:300].split("\n")[:10]))
        print(f"   Explanation: {generation['explanation'][:150]}...")

    # Demonstrate code refactoring
    print(f"\n🔄 Code Refactoring Demo")
    refactoring = await engine.refactor_code(
        sample_codes["complex_python.py"], "Reduce complexity and improve readability"
    )

    if "error" not in refactoring:
        print(f"   Refactoring Improvements: {refactoring['improvements'][:200]}...")
        print(f"   Rationale: {refactoring['rationale'][:150]}...")

    # Demonstrate documentation generation
    print(f"\n📚 Documentation Generation Demo")
    documentation = await engine.generate_documentation(
        sample_codes["python_example.py"]
    )

    if "error" not in documentation:
        print(f"   Generated Documentation: {documentation['documentation'][:200]}...")

    # Demonstrate code review
    print(f"\n👀 Code Review Demo")
    review = await engine.review_code(sample_codes["javascript_example.js"])

    if "error" not in review:
        print(f"   Review Summary: {review['review_summary'][:150]}...")
        print(f"   Strengths: {review['strengths'][:100]}...")
        print(f"   Recommendations: {review['recommendations'][:150]}...")

    # List all analyses
    print(f"\n📋 Analysis Summary")
    analyses = engine.list_analyses()
    for analysis in analyses:
        print(
            f"   {analysis['file_path']}: {analysis['language']} "
            f"(Quality: {analysis['quality_score']:.2f}, Issues: {analysis['issue_count']})"
        )

    print(f"\n✅ Code Analysis Demo Complete!")


async def test_code_analysis_system():
    """Test the code analysis system"""
    print("🧪 Testing Code Analysis System")
    print("=" * 50)

    # Test language detection
    detector = CodeLanguageDetector()

    test_codes = {
        "def hello(): pass": CodeLanguage.PYTHON,
        "function hello() {}": CodeLanguage.JAVASCRIPT,
        "public class Hello {}": CodeLanguage.JAVA,
        "#include <stdio.h>": CodeLanguage.C,
    }

    print("Testing language detection:")
    for code, expected in test_codes.items():
        detected = detector.detect_from_content(code)
        status = "✓" if detected == expected else "✗"
        print(
            f"  {status} '{code[:20]}...' -> {detected.value} (expected: {expected.value})"
        )

    # Test analyzers
    registry = CodeAnalyzerRegistry()
    supported = registry.get_supported_languages()
    print(f"\n✓ Supported languages: {[lang.value for lang in supported]}")

    # Test Python analyzer
    python_analyzer = registry.get_analyzer(CodeLanguage.PYTHON)
    if python_analyzer:
        test_code = "def test():\n    x = 1\n    return x"
        metrics = python_analyzer.calculate_metrics(test_code)
        print(
            f"✓ Python metrics: LOC={metrics.lines_of_code}, Functions={metrics.function_count}"
        )

    # Test engine initialization
    engine = CodeAnalysisEngine()
    print("✓ Code analysis engine initialized")

    # Test API
    api = CodeAnalysisAPI()
    print("✓ API interface initialized")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function to run demos and tests"""
        print("🚀 Advanced Code Analysis and Generation Tool")
        print("=" * 60)

        # Run tests first
        await test_code_analysis_system()
        print()

        # Run demo
        await demo_code_analysis()

        print(f"\n🎯 System Features Demonstrated:")
        print("   ✓ Multi-language code analysis (Python, JavaScript, etc.)")
        print("   ✓ Syntax and style analysis")
        print("   ✓ Code quality metrics calculation")
        print("   ✓ Intelligent issue detection and reporting")
        print("   ✓ AI-powered code explanation")
        print("   ✓ Automated code generation from requirements")
        print("   ✓ Intelligent code refactoring suggestions")
        print("   ✓ Comprehensive documentation generation")
        print("   ✓ Automated code review and recommendations")
        print("   ✓ Extensible analyzer architecture")
        print("   ✓ REST API interface")
        print("   ✓ Comprehensive error handling and logging")

    # Run the main function
    asyncio.run(main())
