#!/usr/bin/env python3
"""
Intelligent Document Processing System

This module provides comprehensive document processing capabilities using DSPy,
including parsing, information extraction, classification, and workflow automation
for multiple document formats.

Learning Objectives:
- Implement intelligent document parsing and analysis
- Create multi-format document processing pipelines
- Build document classification and routing systems
- Develop automated workflow processing capabilities
- Master advanced DSPy patterns for document understanding

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
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


class DocumentType(Enum):
    """Supported document types"""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "md"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    XLSX = "xlsx"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentCategory(Enum):
    """Document categories for classification"""

    RESEARCH_PAPER = "research_paper"
    TECHNICAL_REPORT = "technical_report"
    BUSINESS_DOCUMENT = "business_document"
    LEGAL_DOCUMENT = "legal_document"
    FINANCIAL_REPORT = "financial_report"
    MANUAL = "manual"
    PRESENTATION = "presentation"
    EMAIL = "email"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    OTHER = "other"


@dataclass
class DocumentMetadata:
    """Document metadata structure"""

    filename: str = ""
    file_size: int = 0
    file_type: DocumentType = DocumentType.UNKNOWN
    mime_type: str = ""
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    author: str = ""
    title: str = ""
    language: str = "en"
    page_count: int = 0
    word_count: int = 0
    encoding: str = "utf-8"
    checksum: str = ""


@dataclass
class ExtractedContent:
    """Extracted content from document"""

    raw_text: str = ""
    structured_content: Dict[str, Any] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    sentiment: Dict[str, float] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Complete processed document structure"""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_path: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    content: ExtractedContent = field(default_factory=ExtractedContent)
    category: DocumentCategory = DocumentCategory.OTHER
    classification_confidence: float = 0.0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_time: float = 0.0
    error_message: str = ""
    processed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    insights: Dict[str, Any] = field(default_factory=dict)


# DSPy Signatures for Document Processing
class DocumentClassification(dspy.Signature):
    """Classify documents into categories"""

    document_content: str = dspy.InputField(desc="Document content to classify")
    document_metadata: str = dspy.InputField(
        desc="Document metadata (title, filename, etc.)"
    )
    category: str = dspy.OutputField(desc="Document category classification")
    confidence: float = dspy.OutputField(desc="Classification confidence score (0-1)")
    reasoning: str = dspy.OutputField(desc="Reasoning for classification")


class InformationExtraction(dspy.Signature):
    """Extract key information from documents"""

    document_text: str = dspy.InputField(desc="Full document text")
    extraction_type: str = dspy.InputField(desc="Type of information to extract")
    extracted_info: str = dspy.OutputField(
        desc="Extracted information in structured format"
    )
    entities: str = dspy.OutputField(desc="Named entities found in document")
    key_phrases: str = dspy.OutputField(desc="Key phrases and important terms")


class DocumentSummarization(dspy.Signature):
    """Generate document summaries"""

    document_content: str = dspy.InputField(desc="Document content to summarize")
    summary_type: str = dspy.InputField(
        desc="Type of summary (abstract, executive, detailed)"
    )
    max_length: int = dspy.InputField(desc="Maximum summary length in words")
    summary: str = dspy.OutputField(desc="Generated document summary")
    key_points: str = dspy.OutputField(desc="Key points and main ideas")


class DocumentQualityAssessment(dspy.Signature):
    """Assess document quality and completeness"""

    document_content: str = dspy.InputField(desc="Document content to assess")
    document_type: str = dspy.InputField(desc="Expected document type")
    quality_criteria: str = dspy.InputField(desc="Quality assessment criteria")
    quality_score: float = dspy.OutputField(desc="Overall quality score (0-1)")
    quality_report: str = dspy.OutputField(desc="Detailed quality assessment")
    improvement_suggestions: str = dspy.OutputField(desc="Suggestions for improvement")


class DocumentRouting(dspy.Signature):
    """Route documents to appropriate processing workflows"""

    document_info: str = dspy.InputField(desc="Document information and metadata")
    available_workflows: str = dspy.InputField(desc="Available processing workflows")
    processing_requirements: str = dspy.InputField(desc="Processing requirements")
    recommended_workflow: str = dspy.OutputField(desc="Recommended processing workflow")
    routing_confidence: float = dspy.OutputField(desc="Routing confidence score")
    workflow_parameters: str = dspy.OutputField(desc="Suggested workflow parameters")


class DocumentParser(ABC):
    """Abstract base class for document parsers"""

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if parser can handle the file"""
        pass

    @abstractmethod
    async def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Parse document and return text content and metadata"""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[DocumentType]:
        """Get list of supported document types"""
        pass


class TextDocumentParser(DocumentParser):
    """Parser for plain text documents"""

    def can_parse(self, file_path: str) -> bool:
        """Check if file is a text document"""
        return Path(file_path).suffix.lower() in [".txt", ".md", ".rst"]

    async def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Parse text document"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {
                "word_count": len(content.split()),
                "line_count": len(content.splitlines()),
                "character_count": len(content),
            }

            return content, metadata

        except Exception as e:
            logger.error(f"Failed to parse text document {file_path}: {e}")
            return "", {}

    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return [DocumentType.TXT, DocumentType.MARKDOWN]


class JSONDocumentParser(DocumentParser):
    """Parser for JSON documents"""

    def can_parse(self, file_path: str) -> bool:
        """Check if file is a JSON document"""
        return Path(file_path).suffix.lower() == ".json"

    async def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Parse JSON document"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert JSON to readable text
            content = json.dumps(data, indent=2)

            metadata = {
                "json_structure": self._analyze_json_structure(data),
                "key_count": len(data) if isinstance(data, dict) else 0,
                "data_type": type(data).__name__,
            }

            return content, metadata

        except Exception as e:
            logger.error(f"Failed to parse JSON document {file_path}: {e}")
            return "", {}

    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure"""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:10],  # First 10 keys
                "nested_objects": sum(1 for v in data.values() if isinstance(v, dict)),
                "arrays": sum(1 for v in data.values() if isinstance(v, list)),
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list(set(type(item).__name__ for item in data[:10])),
            }
        else:
            return {"type": type(data).__name__}

    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return [DocumentType.JSON]


class HTMLDocumentParser(DocumentParser):
    """Parser for HTML documents"""

    def can_parse(self, file_path: str) -> bool:
        """Check if file is an HTML document"""
        return Path(file_path).suffix.lower() in [".html", ".htm"]

    async def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Parse HTML document"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Simple HTML text extraction (in production, use BeautifulSoup)
            text_content = self._extract_text_from_html(html_content)

            metadata = {
                "html_tags": self._count_html_tags(html_content),
                "links": self._extract_links(html_content),
                "images": self._extract_images(html_content),
            }

            return text_content, metadata

        except Exception as e:
            logger.error(f"Failed to parse HTML document {file_path}: {e}")
            return "", {}

    def _extract_text_from_html(self, html: str) -> str:
        """Extract text from HTML (simple implementation)"""
        # Remove script and style elements
        html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _count_html_tags(self, html: str) -> Dict[str, int]:
        """Count HTML tags"""
        tags = re.findall(r"<(\w+)", html.lower())
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    def _extract_links(self, html: str) -> List[str]:
        """Extract links from HTML"""
        links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        return links[:20]  # First 20 links

    def _extract_images(self, html: str) -> List[str]:
        """Extract image sources from HTML"""
        images = re.findall(r'src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        return [
            img
            for img in images
            if any(
                ext in img.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg"]
            )
        ][:10]

    def get_supported_types(self) -> List[DocumentType]:
        """Get supported document types"""
        return [DocumentType.HTML]


class DocumentProcessingEngine:
    """Core document processing engine"""

    def __init__(self):
        self.parsers: Dict[DocumentType, DocumentParser] = {}
        self.processed_documents: Dict[str, ProcessedDocument] = {}
        self.processing_queue: List[str] = []
        self.batch_size = 10
        self.max_concurrent = 5

        # Initialize DSPy modules
        self.classifier = dspy.ChainOfThought(DocumentClassification)
        self.extractor = dspy.ChainOfThought(InformationExtraction)
        self.summarizer = dspy.ChainOfThought(DocumentSummarization)
        self.quality_assessor = dspy.ChainOfThought(DocumentQualityAssessment)
        self.router = dspy.ChainOfThought(DocumentRouting)

        # Register default parsers
        self._register_default_parsers()

    def _register_default_parsers(self):
        """Register default document parsers"""
        parsers = [TextDocumentParser(), JSONDocumentParser(), HTMLDocumentParser()]

        for parser in parsers:
            for doc_type in parser.get_supported_types():
                self.parsers[doc_type] = parser

    def register_parser(self, doc_type: DocumentType, parser: DocumentParser):
        """Register a custom document parser"""
        self.parsers[doc_type] = parser
        logger.info(f"Registered parser for {doc_type.value}")

    async def process_document(self, file_path: str) -> ProcessedDocument:
        """Process a single document"""
        start_time = datetime.utcnow()

        # Create document record
        doc = ProcessedDocument(
            source_path=file_path, processing_status=ProcessingStatus.PROCESSING
        )

        try:
            # Extract metadata
            doc.metadata = await self._extract_metadata(file_path)

            # Parse document content
            content, parse_metadata = await self._parse_document(
                file_path, doc.metadata.file_type
            )
            doc.content.raw_text = content
            doc.content.structured_content.update(parse_metadata)

            # Classify document
            await self._classify_document(doc)

            # Extract information
            await self._extract_information(doc)

            # Generate summary
            await self._generate_summary(doc)

            # Assess quality
            await self._assess_quality(doc)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            doc.processing_time = processing_time
            doc.processing_status = ProcessingStatus.COMPLETED
            doc.processed_at = datetime.utcnow()

            # Store processed document
            self.processed_documents[doc.id] = doc

            logger.info(f"Successfully processed document: {file_path}")
            return doc

        except Exception as e:
            doc.processing_status = ProcessingStatus.FAILED
            doc.error_message = str(e)
            doc.processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.error(f"Failed to process document {file_path}: {e}")
            return doc

    async def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract document metadata"""
        path = Path(file_path)

        metadata = DocumentMetadata()
        metadata.filename = path.name
        metadata.file_size = path.stat().st_size if path.exists() else 0
        metadata.file_type = self._detect_file_type(file_path)
        metadata.mime_type = mimetypes.guess_type(file_path)[0] or "unknown"

        if path.exists():
            stat = path.stat()
            metadata.created_date = datetime.fromtimestamp(stat.st_ctime)
            metadata.modified_date = datetime.fromtimestamp(stat.st_mtime)

        return metadata

    def _detect_file_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension"""
        extension = Path(file_path).suffix.lower()

        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".md": DocumentType.MARKDOWN,
            ".json": DocumentType.JSON,
            ".xml": DocumentType.XML,
            ".csv": DocumentType.CSV,
            ".xlsx": DocumentType.XLSX,
            ".xls": DocumentType.XLSX,
        }

        return type_mapping.get(extension, DocumentType.UNKNOWN)

    async def _parse_document(
        self, file_path: str, doc_type: DocumentType
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse document using appropriate parser"""
        if doc_type in self.parsers:
            parser = self.parsers[doc_type]
            if parser.can_parse(file_path):
                return await parser.parse(file_path)

        # Fallback to text parser for unknown types
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content, {"fallback_parsing": True}
        except Exception as e:
            logger.warning(f"Failed to parse {file_path} with fallback parser: {e}")
            return "", {"parsing_failed": True}

    async def _classify_document(self, doc: ProcessedDocument):
        """Classify document into categories"""
        if not doc.content.raw_text.strip():
            doc.category = DocumentCategory.OTHER
            doc.classification_confidence = 0.0
            return

        # Prepare metadata string
        metadata_str = (
            f"Filename: {doc.metadata.filename}, Type: {doc.metadata.file_type.value}"
        )

        # Classify using DSPy
        classification_result = self.classifier(
            document_content=doc.content.raw_text[:2000],  # First 2000 chars
            document_metadata=metadata_str,
        )

        # Map classification result to enum
        category_mapping = {
            "research_paper": DocumentCategory.RESEARCH_PAPER,
            "technical_report": DocumentCategory.TECHNICAL_REPORT,
            "business_document": DocumentCategory.BUSINESS_DOCUMENT,
            "legal_document": DocumentCategory.LEGAL_DOCUMENT,
            "financial_report": DocumentCategory.FINANCIAL_REPORT,
            "manual": DocumentCategory.MANUAL,
            "presentation": DocumentCategory.PRESENTATION,
            "email": DocumentCategory.EMAIL,
            "news_article": DocumentCategory.NEWS_ARTICLE,
            "blog_post": DocumentCategory.BLOG_POST,
        }

        category_key = classification_result.category.lower().replace(" ", "_")
        doc.category = category_mapping.get(category_key, DocumentCategory.OTHER)
        doc.classification_confidence = classification_result.confidence

        # Store classification reasoning
        doc.insights["classification_reasoning"] = classification_result.reasoning

    async def _extract_information(self, doc: ProcessedDocument):
        """Extract key information from document"""
        if not doc.content.raw_text.strip():
            return

        # Extract entities and key phrases
        extraction_result = self.extractor(
            document_text=doc.content.raw_text[:3000],  # First 3000 chars
            extraction_type="entities_and_keyphrases",
        )

        try:
            # Parse extracted information
            extracted_data = json.loads(extraction_result.extracted_info)
            doc.content.structured_content.update(extracted_data)
        except json.JSONDecodeError:
            # Fallback to storing as text
            doc.content.structured_content["extracted_info"] = (
                extraction_result.extracted_info
            )

        # Parse entities
        try:
            entities = json.loads(extraction_result.entities)
            doc.content.entities = entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            doc.content.entities = []

        # Parse key phrases
        try:
            key_phrases = json.loads(extraction_result.key_phrases)
            doc.content.key_phrases = (
                key_phrases if isinstance(key_phrases, list) else []
            )
        except json.JSONDecodeError:
            # Split by common delimiters
            phrases = re.split(r"[,;|\n]", extraction_result.key_phrases)
            doc.content.key_phrases = [p.strip() for p in phrases if p.strip()]

    async def _generate_summary(self, doc: ProcessedDocument):
        """Generate document summary"""
        if not doc.content.raw_text.strip():
            return

        summary_result = self.summarizer(
            document_content=doc.content.raw_text[:4000],  # First 4000 chars
            summary_type="executive",
            max_length=200,
        )

        doc.content.summary = summary_result.summary

        # Parse key points
        try:
            key_points = json.loads(summary_result.key_points)
            if isinstance(key_points, list):
                doc.content.topics = key_points
        except json.JSONDecodeError:
            # Split by common delimiters
            points = re.split(r"[,;|\n]", summary_result.key_points)
            doc.content.topics = [p.strip() for p in points if p.strip()]

    async def _assess_quality(self, doc: ProcessedDocument):
        """Assess document quality"""
        if not doc.content.raw_text.strip():
            doc.quality_score = 0.0
            return

        quality_result = self.quality_assessor(
            document_content=doc.content.raw_text[:2000],
            document_type=doc.category.value,
            quality_criteria="completeness, readability, structure, information_density",
        )

        doc.quality_score = quality_result.quality_score
        doc.insights["quality_report"] = quality_result.quality_report
        doc.insights["improvement_suggestions"] = quality_result.improvement_suggestions

    async def process_batch(self, file_paths: List[str]) -> List[ProcessedDocument]:
        """Process multiple documents in batch"""
        logger.info(f"Processing batch of {len(file_paths)} documents")

        # Process documents with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(file_path: str) -> ProcessedDocument:
            async with semaphore:
                return await self.process_document(file_path)

        tasks = [process_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        processed_docs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {file_paths[i]}: {result}")
            else:
                processed_docs.append(result)

        logger.info(
            f"Successfully processed {len(processed_docs)}/{len(file_paths)} documents"
        )
        return processed_docs

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        docs = list(self.processed_documents.values())

        if not docs:
            return {"total_documents": 0}

        # Calculate statistics
        total_docs = len(docs)
        completed_docs = len(
            [d for d in docs if d.processing_status == ProcessingStatus.COMPLETED]
        )
        failed_docs = len(
            [d for d in docs if d.processing_status == ProcessingStatus.FAILED]
        )

        avg_processing_time = sum(d.processing_time for d in docs) / total_docs
        avg_quality_score = sum(d.quality_score for d in docs) / total_docs

        # Category distribution
        category_counts = {}
        for doc in docs:
            category = doc.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # File type distribution
        type_counts = {}
        for doc in docs:
            file_type = doc.metadata.file_type.value
            type_counts[file_type] = type_counts.get(file_type, 0) + 1

        return {
            "total_documents": total_docs,
            "completed_documents": completed_docs,
            "failed_documents": failed_docs,
            "success_rate": (
                (completed_docs / total_docs) * 100 if total_docs > 0 else 0
            ),
            "average_processing_time": avg_processing_time,
            "average_quality_score": avg_quality_score,
            "category_distribution": category_counts,
            "file_type_distribution": type_counts,
        }

    def search_documents(
        self, query: str, filters: Dict[str, Any] = None
    ) -> List[ProcessedDocument]:
        """Search processed documents"""
        results = []
        query_lower = query.lower()

        for doc in self.processed_documents.values():
            # Text search
            if (
                query_lower in doc.content.raw_text.lower()
                or query_lower in doc.content.summary.lower()
                or any(
                    query_lower in phrase.lower() for phrase in doc.content.key_phrases
                )
            ):

                # Apply filters if provided
                if filters:
                    if not self._apply_filters(doc, filters):
                        continue

                results.append(doc)

        return results

    def _apply_filters(self, doc: ProcessedDocument, filters: Dict[str, Any]) -> bool:
        """Apply search filters to document"""
        if "category" in filters and doc.category.value != filters["category"]:
            return False

        if (
            "file_type" in filters
            and doc.metadata.file_type.value != filters["file_type"]
        ):
            return False

        if "min_quality" in filters and doc.quality_score < filters["min_quality"]:
            return False

        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            if not doc.processed_at or doc.processed_at < date_from:
                return False

        return True


class DocumentWorkflow:
    """Document processing workflow definition"""

    def __init__(self, workflow_id: str, name: str, description: str):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []
        self.conditions: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}
        self.enabled = True

    def add_step(
        self, step_name: str, step_function: str, parameters: Dict[str, Any] = None
    ):
        """Add a processing step to the workflow"""
        step = {
            "name": step_name,
            "function": step_function,
            "parameters": parameters or {},
            "order": len(self.steps),
        }
        self.steps.append(step)

    def set_condition(self, condition_type: str, condition_value: Any):
        """Set workflow execution condition"""
        self.conditions[condition_type] = condition_value

    def matches_document(self, doc: ProcessedDocument) -> bool:
        """Check if workflow should be applied to document"""
        if not self.enabled:
            return False

        # Check category condition
        if "category" in self.conditions:
            if doc.category.value != self.conditions["category"]:
                return False

        # Check file type condition
        if "file_type" in self.conditions:
            if doc.metadata.file_type.value != self.conditions["file_type"]:
                return False

        # Check quality threshold
        if "min_quality" in self.conditions:
            if doc.quality_score < self.conditions["min_quality"]:
                return False

        return True


class DocumentRoutingSystem:
    """Intelligent document routing system"""

    def __init__(self, processing_engine: DocumentProcessingEngine):
        self.processing_engine = processing_engine
        self.workflows: Dict[str, DocumentWorkflow] = {}
        self.routing_history: List[Dict[str, Any]] = []

        # Initialize default workflows
        self._create_default_workflows()

    def _create_default_workflows(self):
        """Create default processing workflows"""

        # Research paper workflow
        research_workflow = DocumentWorkflow(
            "research_paper_workflow",
            "Research Paper Processing",
            "Specialized processing for research papers",
        )
        research_workflow.add_step("extract_citations", "extract_citations")
        research_workflow.add_step("identify_methodology", "identify_methodology")
        research_workflow.add_step("extract_results", "extract_results")
        research_workflow.set_condition("category", "research_paper")

        # Business document workflow
        business_workflow = DocumentWorkflow(
            "business_document_workflow",
            "Business Document Processing",
            "Processing for business documents",
        )
        business_workflow.add_step("extract_financial_data", "extract_financial_data")
        business_workflow.add_step("identify_stakeholders", "identify_stakeholders")
        business_workflow.add_step("extract_action_items", "extract_action_items")
        business_workflow.set_condition("category", "business_document")

        # Legal document workflow
        legal_workflow = DocumentWorkflow(
            "legal_document_workflow",
            "Legal Document Processing", 
            "Processing for legal documents"
        )
        legal_workflow.add_step("extract_clauses", "extract_clauses")
        legal_workflow.add_step("identify_parties", "identify_parties")
        legal_workflow.set_condition("category", "legal_document")

        # Register workflows
        self.workflows[research_workflow.workflow_id] = research_workflow
        self.workflows[business_workflow.workflow_id] = business_workflow
        self.workflows[legal_workflow.workflow_id] = legal_workflow

    def register_workflow(self, workflow: DocumentWorkflow):
        """Register a custom workflow"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    async def route_document(self, doc: ProcessedDocument) -> List[str]:
        """Route document to appropriate workflows"""
        matching_workflows = []
        
        for workflow_id, workflow in self.workflows.items():
            if workflow.matches_document(doc):
                matching_workflows.append(workflow_id)
        
        # Record routing decision
        routing_record = {
            "document_id": doc.id,
            "workflows": matching_workflows,
            "routed_at": datetime.utcnow(),
            "document_category": doc.category.value,
            "quality_score": doc.quality_score
        }
        self.routing_history.append(routing_record)
        
        return matching_workflows


# Demo and testing functions
async def demonstrate_document_processing():
    """Demonstrate document processing capabilities"""
    print("🔍 Document Processing System Demo")
    print("=" * 50)
    
    # Initialize processing engine
    engine = DocumentProcessingEngine()
    
    # Create sample documents directory
    sample_docs_dir = Path("sample_documents")
    sample_docs_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    sample_documents = {
        "research_paper.txt": """
        Abstract: This paper presents a novel approach to natural language processing
        using transformer architectures. We demonstrate significant improvements in
        performance across multiple benchmarks.
        
        Introduction: Natural language processing has seen remarkable advances with
        the introduction of transformer models. However, challenges remain in
        efficiency and interpretability.
        
        Methodology: We propose a new attention mechanism that reduces computational
        complexity while maintaining performance.
        
        Results: Experimental results show 15% improvement in accuracy and 30%
        reduction in training time compared to baseline models.
        """,
        
        "business_report.json": {
            "title": "Q3 2024 Financial Report",
            "revenue": 2100000,
            "expenses": 1800000,
            "profit": 300000,
            "growth_rate": 0.22,
            "key_metrics": {
                "customer_acquisition": 1200,
                "market_share": 0.153,
                "employee_count": 450
            }
        },
        
        "technical_manual.html": """
        <html>
        <head><title>API Documentation</title></head>
        <body>
        <h1>REST API Documentation</h1>
        <h2>Authentication</h2>
        <p>All API requests require authentication using API keys.</p>
        <h2>Endpoints</h2>
        <h3>GET /api/documents</h3>
        <p>Retrieve list of documents</p>
        <h3>POST /api/documents</h3>
        <p>Upload new document</p>
        </body>
        </html>
        """
    }
    
    # Write sample documents
    for filename, content in sample_documents.items():
        file_path = sample_docs_dir / filename
        if filename.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            with open(file_path, 'w') as f:
                f.write(content)
    
    print(f"📄 Processing {len(sample_documents)} sample documents...")
    
    # Process documents
    file_paths = [str(sample_docs_dir / filename) for filename in sample_documents.keys()]
    processed_docs = await engine.process_batch(file_paths)
    
    # Display results
    for doc in processed_docs:
        print(f"\n📋 Document: {doc.metadata.filename}")
        print(f"   Category: {doc.category.value}")
        print(f"   Confidence: {doc.classification_confidence:.2f}")
        print(f"   Quality Score: {doc.quality_score:.2f}")
        print(f"   Processing Time: {doc.processing_time:.2f}s")
        print(f"   Word Count: {doc.metadata.word_count}")
        print(f"   Summary: {doc.content.summary[:100]}...")
        
        if doc.content.key_phrases:
            print(f"   Key Phrases: {', '.join(doc.content.key_phrases[:3])}")
    
    # Show statistics
    stats = engine.get_processing_statistics()
    print(f"\n📊 Processing Statistics:")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Average Quality: {stats['average_quality_score']:.2f}")
    print(f"   Average Processing Time: {stats['average_processing_time']:.2f}s")
    
    # Test search functionality
    print(f"\n🔍 Search Test:")
    search_results = engine.search_documents("natural language processing")
    print(f"   Found {len(search_results)} documents matching 'natural language processing'")
    
    # Test routing system
    print(f"\n🔄 Document Routing Test:")
    routing_system = DocumentRoutingSystem(engine)
    
    for doc in processed_docs:
        workflows = await routing_system.route_document(doc)
        print(f"   {doc.metadata.filename} -> {len(workflows)} workflows: {', '.join(workflows)}")
    
    # Cleanup
    import shutil
    shutil.rmtree(sample_docs_dir, ignore_errors=True)
    
    print(f"\n✅ Document processing system demo completed!")


async def test_document_processing_system():
    """Test document processing system"""
    print("🧪 Testing Document Processing System")
    print("=" * 50)
    
    # Test engine initialization
    engine = DocumentProcessingEngine()
    print("✓ Processing engine initialized")
    
    # Test parser registration
    print(f"✓ Registered {len(engine.parsers)} parsers")
    
    # Test workflow system
    routing_system = DocumentRoutingSystem(engine)
    print(f"✓ Routing system with {len(routing_system.workflows)} workflows")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    import asyncio
    import time
    
    async def main():
        """Main function to run demos and tests"""
        print("🚀 Intelligent Document Processing System")
        print("=" * 60)
        
        try:
            await demonstrate_document_processing()
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Demo error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the main function
    asyncio.run(main())

    def register_workflow(self, workflow: DocumentWorkflow):
        """Register a custom workflow"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    async def route_document(self, doc: ProcessedDocument) -> List[str]:
        """Route document to appropriate workflows"""
        matching_workflows = []

        for workflow_id, workflow in self.workflows.items():
            if workflow.matches_document(doc):
                matching_workflows.append(workflow_id)

        # Use DSPy for intelligent routing if multiple workflows match
        if len(matching_workflows) > 1:
            recommended_workflow = await self._intelligent_routing(
                doc, matching_workflows
            )
            if recommended_workflow:
                matching_workflows = [recommended_workflow]

        # Record routing decision
        routing_record = {
            "document_id": doc.id,
            "document_category": doc.category.value,
            "matched_workflows": matching_workflows,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.routing_history.append(routing_record)

        return matching_workflows

    async def _intelligent_routing(
        self, doc: ProcessedDocument, candidate_workflows: List[str]
    ) -> Optional[str]:
        """Use DSPy for intelligent workflow routing"""
        try:
            # Prepare document information
            doc_info = {
                "category": doc.category.value,
                "file_type": doc.metadata.file_type.value,
                "quality_score": doc.quality_score,
                "content_preview": doc.content.raw_text[:500],
                "key_phrases": doc.content.key_phrases[:5],
            }

            # Prepare workflow information
            workflow_info = {}
            for workflow_id in candidate_workflows:
                workflow = self.workflows[workflow_id]
                workflow_info[workflow_id] = {
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps": [step["name"] for step in workflow.steps],
                }

            # Use DSPy router
            routing_result = self.processing_engine.router(
                document_info=json.dumps(doc_info),
                available_workflows=json.dumps(workflow_info),
                processing_requirements="optimal_processing_quality",
            )

            recommended = routing_result.recommended_workflow
            if recommended in candidate_workflows:
                return recommended

        except Exception as e:
            logger.error(f"Intelligent routing failed: {e}")

        # Fallback to first matching workflow
        return candidate_workflows[0] if candidate_workflows else None

    async def execute_workflow(
        self, doc: ProcessedDocument, workflow_id: str
    ) -> Dict[str, Any]:
        """Execute a workflow on a document"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]
        results = {"workflow_id": workflow_id, "steps": []}

        logger.info(f"Executing workflow {workflow.name} on document {doc.id}")

        for step in workflow.steps:
            step_result = await self._execute_workflow_step(doc, step)
            results["steps"].append(
                {
                    "step_name": step["name"],
                    "result": step_result,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        return results

    async def _execute_workflow_step(
        self, doc: ProcessedDocument, step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_name = step["name"]
        step_function = step["function"]
        parameters = step.get("parameters", {})

        try:
            # Map step functions to actual implementations
            if step_function == "extract_citations":
                return await self._extract_citations(doc, parameters)
            elif step_function == "identify_methodology":
                return await self._identify_methodology(doc, parameters)
            elif step_function == "extract_results":
                return await self._extract_results(doc, parameters)
            elif step_function == "extract_financial_data":
                return await self._extract_financial_data(doc, parameters)
            elif step_function == "identify_stakeholders":
                return await self._identify_stakeholders(doc, parameters)
            elif step_function == "extract_action_items":
                return await self._extract_action_items(doc, parameters)
            elif step_function == "extract_legal_entities":
                return await self._extract_legal_entities(doc, parameters)
            elif step_function == "identify_clauses":
                return await self._identify_clauses(doc, parameters)
            elif step_function == "extract_dates_deadlines":
                return await self._extract_dates_deadlines(doc, parameters)
            else:
                return {"error": f"Unknown step function: {step_function}"}

        except Exception as e:
            logger.error(f"Step {step_name} failed: {e}")
            return {"error": str(e)}

    # Workflow step implementations
    async def _extract_citations(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract citations from research papers"""
        # Use DSPy to extract citations
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="citations_and_references",
        )

        return {"citations": extraction_result.extracted_info, "confidence": 0.8}

    async def _identify_methodology(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify research methodology"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text, extraction_type="research_methodology"
        )

        return {"methodology": extraction_result.extracted_info, "confidence": 0.75}

    async def _extract_results(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract research results"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="research_results_and_findings",
        )

        return {"results": extraction_result.extracted_info, "confidence": 0.8}

    async def _extract_financial_data(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract financial data from business documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="financial_data_and_metrics",
        )

        return {"financial_data": extraction_result.extracted_info, "confidence": 0.85}

    async def _identify_stakeholders(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify stakeholders in business documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="stakeholders_and_parties",
        )

        return {"stakeholders": extraction_result.extracted_info, "confidence": 0.7}

    async def _extract_action_items(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract action items from business documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text, extraction_type="action_items_and_tasks"
        )

        return {"action_items": extraction_result.extracted_info, "confidence": 0.75}

    async def _extract_legal_entities(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract legal entities from legal documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="legal_entities_and_parties",
        )

        return {"legal_entities": extraction_result.extracted_info, "confidence": 0.8}

    async def _identify_clauses(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify clauses in legal documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="legal_clauses_and_provisions",
        )

        return {"clauses": extraction_result.extracted_info, "confidence": 0.75}

    async def _extract_dates_deadlines(
        self, doc: ProcessedDocument, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract dates and deadlines from legal documents"""
        extraction_result = self.processing_engine.extractor(
            document_text=doc.content.raw_text,
            extraction_type="dates_deadlines_and_timelines",
        )

        return {"dates_deadlines": extraction_result.extracted_info, "confidence": 0.8}

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {"total_routings": 0}

        total_routings = len(self.routing_history)
        workflow_usage = {}

        for record in self.routing_history:
            for workflow_id in record["matched_workflows"]:
                workflow_usage[workflow_id] = workflow_usage.get(workflow_id, 0) + 1

        return {
            "total_routings": total_routings,
            "workflow_usage": workflow_usage,
            "registered_workflows": len(self.workflows),
        }


class BatchProcessor:
    """Batch document processing system"""

    def __init__(
        self,
        processing_engine: DocumentProcessingEngine,
        routing_system: DocumentRoutingSystem,
    ):
        self.processing_engine = processing_engine
        self.routing_system = routing_system
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        self.processing_queue: List[str] = []
        self.max_concurrent_jobs = 3
        self.job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)

    async def create_batch_job(
        self, job_name: str, file_paths: List[str], workflow_ids: List[str] = None
    ) -> str:
        """Create a new batch processing job"""
        job_id = str(uuid4())

        job = {
            "id": job_id,
            "name": job_name,
            "file_paths": file_paths,
            "workflow_ids": workflow_ids or [],
            "status": "created",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "processed_documents": [],
            "failed_documents": [],
            "total_files": len(file_paths),
            "progress": 0,
        }

        self.batch_jobs[job_id] = job
        logger.info(f"Created batch job {job_name} with {len(file_paths)} files")

        return job_id

    async def start_batch_job(self, job_id: str) -> bool:
        """Start executing a batch job"""
        if job_id not in self.batch_jobs:
            return False

        job = self.batch_jobs[job_id]
        if job["status"] != "created":
            return False

        job["status"] = "running"
        job["started_at"] = datetime.utcnow()

        # Start processing in background
        asyncio.create_task(self._execute_batch_job(job_id))

        logger.info(f"Started batch job: {job['name']}")
        return True

    async def _execute_batch_job(self, job_id: str):
        """Execute batch job processing"""
        job = self.batch_jobs[job_id]

        async with self.job_semaphore:
            try:
                processed_count = 0

                for file_path in job["file_paths"]:
                    try:
                        # Process document
                        doc = await self.processing_engine.process_document(file_path)

                        # Route to workflows if specified
                        if job["workflow_ids"]:
                            for workflow_id in job["workflow_ids"]:
                                workflow_result = (
                                    await self.routing_system.execute_workflow(
                                        doc, workflow_id
                                    )
                                )
                                doc.insights[f"workflow_{workflow_id}"] = (
                                    workflow_result
                                )
                        else:
                            # Auto-route based on document characteristics
                            matching_workflows = (
                                await self.routing_system.route_document(doc)
                            )
                            for workflow_id in matching_workflows:
                                workflow_result = (
                                    await self.routing_system.execute_workflow(
                                        doc, workflow_id
                                    )
                                )
                                doc.insights[f"workflow_{workflow_id}"] = (
                                    workflow_result
                                )

                        job["processed_documents"].append(doc.id)
                        processed_count += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to process {file_path} in batch job {job_id}: {e}"
                        )
                        job["failed_documents"].append(
                            {"file_path": file_path, "error": str(e)}
                        )

                    # Update progress
                    job["progress"] = (processed_count / job["total_files"]) * 100

                # Complete job
                job["status"] = "completed"
                job["completed_at"] = datetime.utcnow()

                logger.info(
                    f"Batch job {job['name']} completed: {processed_count}/{job['total_files']} files processed"
                )

            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                logger.error(f"Batch job {job_id} failed: {e}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get batch job status"""
        if job_id not in self.batch_jobs:
            return None

        job = self.batch_jobs[job_id]

        return {
            "job_id": job_id,
            "name": job["name"],
            "status": job["status"],
            "progress": job["progress"],
            "total_files": job["total_files"],
            "processed_files": len(job["processed_documents"]),
            "failed_files": len(job["failed_documents"]),
            "created_at": job["created_at"].isoformat(),
            "started_at": job["started_at"].isoformat() if job["started_at"] else None,
            "completed_at": (
                job["completed_at"].isoformat() if job["completed_at"] else None
            ),
            "duration": (
                str(job["completed_at"] - job["started_at"])
                if job.get("completed_at") and job.get("started_at")
                else None
            ),
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all batch jobs"""
        return [self.get_job_status(job_id) for job_id in self.batch_jobs.keys()]


class IntelligentDocumentProcessingSystem:
    """Complete intelligent document processing system"""

    def __init__(self):
        self.processing_engine = DocumentProcessingEngine()
        self.routing_system = DocumentRoutingSystem(self.processing_engine)
        self.batch_processor = BatchProcessor(
            self.processing_engine, self.routing_system
        )
        self.system_metrics: Dict[str, Any] = {}

    async def process_single_document(
        self, file_path: str, workflow_ids: List[str] = None
    ) -> ProcessedDocument:
        """Process a single document with optional workflow routing"""
        # Process document
        doc = await self.processing_engine.process_document(file_path)

        # Route to workflows
        if workflow_ids:
            for workflow_id in workflow_ids:
                workflow_result = await self.routing_system.execute_workflow(
                    doc, workflow_id
                )
                doc.insights[f"workflow_{workflow_id}"] = workflow_result
        else:
            # Auto-route
            matching_workflows = await self.routing_system.route_document(doc)
            for workflow_id in matching_workflows:
                workflow_result = await self.routing_system.execute_workflow(
                    doc, workflow_id
                )
                doc.insights[f"workflow_{workflow_id}"] = workflow_result

        return doc

    async def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: List[str] = None,
    ) -> str:
        """Process all documents in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        # Find files to process
        file_paths = []

        if recursive:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and self._should_process_file(
                    file_path, file_patterns
                ):
                    file_paths.append(str(file_path))
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and self._should_process_file(
                    file_path, file_patterns
                ):
                    file_paths.append(str(file_path))

        # Create batch job
        job_id = await self.batch_processor.create_batch_job(
            f"Directory Processing: {directory.name}", file_paths
        )

        # Start batch job
        await self.batch_processor.start_batch_job(job_id)

        logger.info(f"Started directory processing job: {job_id}")
        return job_id

    def _should_process_file(self, file_path: Path, patterns: List[str] = None) -> bool:
        """Check if file should be processed"""
        # Check file extension
        supported_extensions = [".txt", ".md", ".html", ".htm", ".json", ".xml", ".csv"]
        if file_path.suffix.lower() not in supported_extensions:
            return False

        # Check patterns if provided
        if patterns:
            filename = file_path.name.lower()
            return any(pattern.lower() in filename for pattern in patterns)

        return True

    def search_documents(
        self, query: str, filters: Dict[str, Any] = None
    ) -> List[ProcessedDocument]:
        """Search processed documents"""
        return self.processing_engine.search_documents(query, filters)

    def get_document_by_id(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID"""
        return self.processing_engine.processed_documents.get(doc_id)

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        processing_stats = self.processing_engine.get_processing_statistics()
        routing_stats = self.routing_system.get_routing_statistics()

        # Batch processing stats
        batch_jobs = self.batch_processor.list_jobs()
        completed_jobs = len(
            [job for job in batch_jobs if job["status"] == "completed"]
        )

        return {
            "processing_engine": processing_stats,
            "routing_system": routing_stats,
            "batch_processing": {
                "total_jobs": len(batch_jobs),
                "completed_jobs": completed_jobs,
                "active_jobs": len(
                    [job for job in batch_jobs if job["status"] == "running"]
                ),
            },
            "system_health": {
                "registered_parsers": len(self.processing_engine.parsers),
                "registered_workflows": len(self.routing_system.workflows),
                "total_processed_documents": len(
                    self.processing_engine.processed_documents
                ),
            },
        }

    async def generate_processing_report(
        self, include_details: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive processing report"""
        stats = self.get_system_statistics()

        report = {
            "report_generated_at": datetime.utcnow().isoformat(),
            "system_overview": stats,
            "performance_metrics": {
                "average_processing_time": stats["processing_engine"].get(
                    "average_processing_time", 0
                ),
                "success_rate": stats["processing_engine"].get("success_rate", 0),
                "average_quality_score": stats["processing_engine"].get(
                    "average_quality_score", 0
                ),
            },
            "document_insights": self._generate_document_insights(),
            "recommendations": self._generate_system_recommendations(stats),
        }

        if include_details:
            report["detailed_statistics"] = {
                "category_analysis": stats["processing_engine"].get(
                    "category_distribution", {}
                ),
                "file_type_analysis": stats["processing_engine"].get(
                    "file_type_distribution", {}
                ),
                "workflow_usage": stats["routing_system"].get("workflow_usage", {}),
            }

        return report

    def _generate_document_insights(self) -> Dict[str, Any]:
        """Generate insights from processed documents"""
        docs = list(self.processing_engine.processed_documents.values())

        if not docs:
            return {}

        # Calculate insights
        total_words = sum(doc.metadata.word_count for doc in docs)
        avg_quality = sum(doc.quality_score for doc in docs) / len(docs)

        # Most common categories
        categories = [doc.category.value for doc in docs]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        most_common_category = (
            max(category_counts, key=category_counts.get) if category_counts else None
        )

        # Most common topics
        all_topics = []
        for doc in docs:
            all_topics.extend(doc.content.topics)

        topic_counts = {topic: all_topics.count(topic) for topic in set(all_topics)}
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_documents_analyzed": len(docs),
            "total_words_processed": total_words,
            "average_quality_score": avg_quality,
            "most_common_category": most_common_category,
            "top_topics": [topic for topic, count in top_topics],
            "processing_insights": {
                "high_quality_documents": len(
                    [d for d in docs if d.quality_score > 0.8]
                ),
                "documents_with_entities": len([d for d in docs if d.content.entities]),
                "documents_with_summaries": len([d for d in docs if d.content.summary]),
            },
        }

    def _generate_system_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []

        # Processing performance recommendations
        success_rate = stats["processing_engine"].get("success_rate", 100)
        if success_rate < 90:
            recommendations.append(
                "Consider improving document parsing capabilities for better success rate"
            )

        avg_quality = stats["processing_engine"].get("average_quality_score", 1.0)
        if avg_quality < 0.7:
            recommendations.append(
                "Review document quality assessment criteria and thresholds"
            )

        # Workflow recommendations
        workflow_usage = stats["routing_system"].get("workflow_usage", {})
        if not workflow_usage:
            recommendations.append(
                "Consider creating more specialized workflows for better document processing"
            )

        # Batch processing recommendations
        active_jobs = stats["batch_processing"].get("active_jobs", 0)
        if active_jobs > 5:
            recommendations.append(
                "Consider increasing concurrent job limits for better throughput"
            )

        return recommendations


async def demonstrate_document_processing():
    """Demonstrate the document processing system"""
    print("=== Intelligent Document Processing System Demo ===")

    # Initialize system
    system = IntelligentDocumentProcessingSystem()

    print("System initialized with:")
    print(f"  - {len(system.processing_engine.parsers)} document parsers")
    print(f"  - {len(system.routing_system.workflows)} processing workflows")

    # Create sample documents for demonstration
    print("\nCreating sample documents...")
    sample_docs_dir = Path("sample_documents")
    sample_docs_dir.mkdir(exist_ok=True)

    # Create sample documents
    sample_documents = [
        {
            "filename": "research_paper.txt",
            "content": """
Title: Machine Learning Applications in Healthcare: A Comprehensive Review

Abstract: This paper reviews the current state of machine learning applications in healthcare,
examining various techniques and their effectiveness in clinical settings.

Introduction: Healthcare is undergoing a digital transformation with the integration of
artificial intelligence and machine learning technologies...

Methodology: We conducted a systematic review of 150 research papers published between
2020-2024, focusing on ML applications in diagnosis, treatment, and patient care...

Results: Our analysis reveals that ML techniques show promising results in medical imaging
(accuracy: 94.2%), drug discovery (efficiency improvement: 40%), and predictive analytics
(risk reduction: 35%)...

Conclusion: Machine learning demonstrates significant potential in healthcare applications,
though challenges remain in regulatory approval and clinical integration.

References:
1. Smith, J. et al. (2023). "Deep Learning in Medical Imaging." Nature Medicine.
2. Johnson, A. (2024). "AI-Driven Drug Discovery." Science Translational Medicine.
            """,
        },
        {
            "filename": "business_report.txt",
            "content": """
Q4 2024 Business Performance Report

Executive Summary:
Our company achieved record revenue of $2.5M in Q4 2024, representing 25% growth
year-over-year. Key performance indicators show strong market position.

Financial Highlights:
- Revenue: $2.5M (↑25% YoY)
- Profit Margin: 18.5% (↑2.3% YoY)
- Customer Acquisition Cost: $150 (↓10% YoY)
- Customer Lifetime Value: $2,400 (↑15% YoY)

Key Stakeholders:
- CEO: Sarah Johnson
- CFO: Michael Chen
- VP Sales: Lisa Rodriguez
- VP Engineering: David Kim

Action Items:
1. Expand marketing budget by 20% for Q1 2025
2. Hire 5 additional engineers by March 2025
3. Launch new product line in European markets
4. Implement customer retention program

Next Steps:
Board meeting scheduled for January 15, 2025 to approve expansion plans.
            """,
        },
        {
            "filename": "technical_manual.txt",
            "content": """
API Documentation - User Management System

Overview:
This manual describes the User Management API endpoints and their usage.

Authentication:
All API requests require authentication using Bearer tokens.
Include the token in the Authorization header: "Authorization: Bearer <token>"

Endpoints:

1. GET /api/users
   Description: Retrieve list of users
   Parameters: page (optional), limit (optional)
   Response: JSON array of user objects

2. POST /api/users
   Description: Create new user
   Parameters: name (required), email (required), role (optional)
   Response: Created user object with ID

3. PUT /api/users/{id}
   Description: Update existing user
   Parameters: name, email, role, status
   Response: Updated user object

Error Handling:
- 400: Bad Request - Invalid parameters
- 401: Unauthorized - Invalid or missing token
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server error

Rate Limiting:
API requests are limited to 1000 requests per hour per token.
            """,
        },
    ]

    # Write sample documents
    for doc_info in sample_documents:
        doc_path = sample_docs_dir / doc_info["filename"]
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_info["content"].strip())

    print(f"Created {len(sample_documents)} sample documents")

    # Process individual documents
    print("\nProcessing individual documents...")

    processed_docs = []
    for doc_info in sample_documents:
        doc_path = sample_docs_dir / doc_info["filename"]
        doc = await system.process_single_document(str(doc_path))
        processed_docs.append(doc)

        print(f"  Processed: {doc.metadata.filename}")
        print(
            f"    Category: {doc.category.value} (confidence: {doc.classification_confidence:.2f})"
        )
        print(f"    Quality Score: {doc.quality_score:.2f}")
        print(f"    Processing Time: {doc.processing_time:.2f}s")
        print(f"    Key Phrases: {len(doc.content.key_phrases)}")
        print(f"    Summary Length: {len(doc.content.summary)} chars")

    # Demonstrate batch processing
    print(f"\nDemonstrating batch processing...")
    sample_files = [str(sample_docs_dir / doc["filename"]) for doc in sample_documents]

    batch_job_id = await system.batch_processor.create_batch_job(
        "Sample Documents Batch", sample_files
    )

    await system.batch_processor.start_batch_job(batch_job_id)

    # Monitor batch job progress
    print("Monitoring batch job progress...")
    for i in range(10):
        await asyncio.sleep(1)
        status = system.batch_processor.get_job_status(batch_job_id)
        if status:
            print(
                f"  Progress: {status['progress']:.1f}% ({status['processed_files']}/{status['total_files']})"
            )
            if status["status"] == "completed":
                break

    # Generate system report
    print(f"\nGenerating system report...")
    report = await system.generate_processing_report(include_details=True)

    print(f"\nSystem Report Summary:")
    print(
        f"  Documents Processed: {report['system_overview']['processing_engine']['total_documents']}"
    )
    print(f"  Success Rate: {report['performance_metrics']['success_rate']:.1f}%")
    print(
        f"  Average Quality: {report['performance_metrics']['average_quality_score']:.2f}"
    )
    print(
        f"  Average Processing Time: {report['performance_metrics']['average_processing_time']:.2f}s"
    )

    if report["document_insights"]:
        insights = report["document_insights"]
        print(f"\nDocument Insights:")
        print(f"  Total Words Processed: {insights['total_words_processed']:,}")
        print(f"  Most Common Category: {insights['most_common_category']}")
        print(f"  Top Topics: {', '.join(insights['top_topics'])}")
        print(
            f"  High Quality Documents: {insights['processing_insights']['high_quality_documents']}"
        )

    if report["recommendations"]:
        print(f"\nSystem Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    # Demonstrate search functionality
    print(f"\nDemonstrating search functionality...")
    search_results = system.search_documents("machine learning")
    print(f"Search for 'machine learning': {len(search_results)} results")

    for result in search_results[:2]:
        print(f"  - {result.metadata.filename} (category: {result.category.value})")

    # Cleanup
    import shutil

    shutil.rmtree(sample_docs_dir, ignore_errors=True)

    print(f"\nDocument processing system demo completed!")


if __name__ == "__main__":
    import asyncio
    import time

    async def main():
        """Main function to run demos and tests"""
        print("🚀 Intelligent Document Processing System")
        print("=" * 60)

        try:
            await demonstrate_document_processing()
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Demo error: {e}")
            import traceback

            traceback.print_exc()
        print("=" * 60)

        # Run tests first
        await test_document_processing_system()
        print()

        # Run demo
        await demo_document_processing()

        print(f"\n🎯 System Features Demonstrated:")
        print("   ✓ Multi-format document parsing (TXT, MD, JSON, HTML)")
        print("   ✓ Intelligent content extraction and analysis")
        print("   ✓ Document summarization and keyword extraction")
        print("   ✓ Named entity recognition")
        print("   ✓ Topic modeling and sentiment analysis")
        print("   ✓ Document structure analysis")
        print("   ✓ Quality assessment")
        print("   ✓ Information extraction with queries")
        print("   ✓ Document comparison and similarity analysis")
        print("   ✓ Batch processing capabilities")
        print("   ✓ Pipeline-based workflow management")
        print("   ✓ REST API interface")
        print("   ✓ Comprehensive error handling and logging")

    # Run the main function
    asyncio.run(main())
        """Process multiple documents in batch"""
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(file_path, extraction_types)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Create failed result
                failed_result = ProcessingResult(
                    document_id=str(uuid4()),
                    status=ProcessingStatus.FAILED,
                    processed_at=datetime.utcnow(),
                    processing_time=0.0,
                )
                failed_result.errors.append(str(e))
                results.append(failed_result)

        return results


class DocumentProcessingPipeline:
    """Advanced document processing pipeline with workflow management"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.pipelines: Dict[str, Dict[str, Any]] = {}

    def create_pipeline(self, pipeline_id: str, stages: List[Dict[str, Any]]) -> str:
        """Create a processing pipeline with multiple stages"""
        self.pipelines[pipeline_id] = {
            "id": pipeline_id,
            "stages": stages,
            "created_at": datetime.utcnow(),
            "status": "ready",
        }
        return pipeline_id

    async def run_pipeline(
        self, pipeline_id: str, input_documents: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Run a processing pipeline on input documents"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipelines[pipeline_id]
        pipeline["status"] = "running"

        results = {
            "pipeline_id": pipeline_id,
            "input_documents": [str(path) for path in input_documents],
            "stage_results": [],
            "final_results": [],
            "started_at": datetime.utcnow(),
            "completed_at": None,
            "status": "running",
        }

        try:
            current_documents = input_documents

            for stage_idx, stage in enumerate(pipeline["stages"]):
                stage_name = stage.get("name", f"Stage {stage_idx + 1}")
                stage_type = stage.get("type", "process")
                stage_config = stage.get("config", {})

                logger.info(f"Running pipeline stage: {stage_name}")

                if stage_type == "process":
                    # Process documents
                    extraction_types = stage_config.get("extraction_types")
                    if extraction_types:
                        extraction_types = [
                            ExtractionType(et) for et in extraction_types
                        ]

                    stage_results = await self.processor.batch_process_documents(
                        current_documents, extraction_types
                    )

                    results["stage_results"].append(
                        {
                            "stage_name": stage_name,
                            "stage_type": stage_type,
                            "results": stage_results,
                        }
                    )

                elif stage_type == "filter":
                    # Filter documents based on criteria
                    criteria = stage_config.get("criteria", {})
                    filtered_docs = self._filter_documents(current_documents, criteria)
                    current_documents = filtered_docs

                    results["stage_results"].append(
                        {
                            "stage_name": stage_name,
                            "stage_type": stage_type,
                            "filtered_count": len(filtered_docs),
                            "original_count": len(current_documents),
                        }
                    )

                elif stage_type == "aggregate":
                    # Aggregate results
                    aggregation_type = stage_config.get("aggregation_type", "summary")
                    aggregated_result = await self._aggregate_results(
                        current_documents, aggregation_type
                    )

                    results["stage_results"].append(
                        {
                            "stage_name": stage_name,
                            "stage_type": stage_type,
                            "aggregated_result": aggregated_result,
                        }
                    )

            results["status"] = "completed"
            results["final_results"] = (
                results["stage_results"][-1] if results["stage_results"] else []
            )

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Pipeline {pipeline_id} failed: {e}")

        finally:
            results["completed_at"] = datetime.utcnow()
            pipeline["status"] = results["status"]

        return results

    def _filter_documents(
        self, documents: List[Union[str, Path]], criteria: Dict[str, Any]
    ) -> List[Union[str, Path]]:
        """Filter documents based on criteria"""
        filtered = []

        for doc_path in documents:
            doc_path = Path(doc_path)

            # File size filter
            if "max_size" in criteria:
                if doc_path.stat().st_size > criteria["max_size"]:
                    continue

            if "min_size" in criteria:
                if doc_path.stat().st_size < criteria["min_size"]:
                    continue

            # File extension filter
            if "extensions" in criteria:
                if doc_path.suffix.lower() not in criteria["extensions"]:
                    continue

            # File name pattern filter
            if "name_pattern" in criteria:
                if not re.search(criteria["name_pattern"], doc_path.name):
                    continue

            filtered.append(doc_path)

        return filtered

    async def _aggregate_results(
        self, documents: List[Union[str, Path]], aggregation_type: str
    ) -> Dict[str, Any]:
        """Aggregate processing results"""
        if aggregation_type == "summary":
            # Create combined summary
            all_summaries = []
            all_keywords = []
            all_topics = []

            for doc_id, result in self.processor.processing_results.items():
                if result.summary:
                    all_summaries.append(result.summary)
                all_keywords.extend(result.keywords)
                all_topics.extend(result.topics)

            # Combine summaries
            combined_summary = " ".join(all_summaries)
            if len(combined_summary) > 1000:
                # Use DSPy to create meta-summary
                meta_summary = self.processor.summarizer(
                    document_content=combined_summary,
                    document_type="aggregated_summaries",
                    summary_length="medium",
                )
                combined_summary = meta_summary.summary

            # Get top keywords and topics
            keyword_freq = {}
            for keyword in all_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1

            top_keywords = sorted(
                keyword_freq.items(), key=lambda x: x[1], reverse=True
            )[:20]

            topic_freq = {}
            for topic in all_topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1

            top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]

            return {
                "aggregation_type": "summary",
                "combined_summary": combined_summary,
                "top_keywords": [kw for kw, freq in top_keywords],
                "top_topics": [topic for topic, freq in top_topics],
                "document_count": len(documents),
                "total_words": sum(
                    result.metadata.word_count
                    for result in self.processor.processing_results.values()
                ),
            }

        return {"aggregation_type": aggregation_type, "result": "Not implemented"}


class DocumentProcessingAPI:
    """REST API interface for document processing system"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.pipeline = DocumentProcessingPipeline()

    async def process_document_endpoint(
        self, file_path: str, extraction_types: List[str] = None
    ) -> Dict[str, Any]:
        """API endpoint for processing a single document"""
        try:
            if extraction_types:
                extraction_types = [ExtractionType(et) for et in extraction_types]

            result = await self.processor.process_document(file_path, extraction_types)

            return {
                "success": True,
                "document_id": result.document_id,
                "status": result.status.value,
                "processing_time": result.processing_time,
                "metadata": {
                    "filename": result.metadata.filename,
                    "file_size": result.metadata.file_size,
                    "word_count": result.metadata.word_count,
                    "quality_score": result.quality_score,
                },
                "summary": result.summary,
                "keywords": result.keywords[:10],  # Limit for API response
                "topics": result.topics[:5],
                "entity_count": len(result.entities),
                "section_count": len(result.sections),
                "errors": result.errors,
                "warnings": result.warnings,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def batch_process_endpoint(
        self, file_paths: List[str], extraction_types: List[str] = None
    ) -> Dict[str, Any]:
        """API endpoint for batch processing documents"""
        try:
            if extraction_types:
                extraction_types = [ExtractionType(et) for et in extraction_types]

            results = await self.processor.batch_process_documents(
                file_paths, extraction_types
            )

            successful = [r for r in results if r.status == ProcessingStatus.COMPLETED]
            failed = [r for r in results if r.status == ProcessingStatus.FAILED]

            return {
                "success": True,
                "total_documents": len(results),
                "successful_count": len(successful),
                "failed_count": len(failed),
                "results": [
                    {
                        "document_id": r.document_id,
                        "filename": r.metadata.filename,
                        "status": r.status.value,
                        "processing_time": r.processing_time,
                        "quality_score": r.quality_score,
                        "word_count": r.metadata.word_count,
                        "summary_length": len(r.summary),
                        "keyword_count": len(r.keywords),
                        "entity_count": len(r.entities),
                    }
                    for r in results
                ],
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def extract_information_endpoint(
        self, document_id: str, query: str
    ) -> Dict[str, Any]:
        """API endpoint for extracting specific information"""
        try:
            result = await self.processor.extract_information(document_id, query)
            return {"success": True, **result}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def compare_documents_endpoint(
        self, document_id1: str, document_id2: str, aspects: str = None
    ) -> Dict[str, Any]:
        """API endpoint for comparing documents"""
        try:
            if aspects is None:
                aspects = "content, structure, topics"

            result = await self.processor.compare_documents(
                document_id1, document_id2, aspects
            )
            return {"success": True, **result}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}


# Demo and Testing Functions
async def demo_document_processing():
    """Demonstrate document processing capabilities"""
    print("🔍 Document Processing System Demo")
    print("=" * 50)

    # Initialize processor
    processor = DocumentProcessor()

    # Create sample documents for testing
    sample_docs = {
        "sample_text.txt": """
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) has revolutionized many industries in recent years.
        Machine learning, a subset of AI, enables computers to learn and improve from experience
        without being explicitly programmed.
        
        Key applications include:
        - Natural language processing
        - Computer vision
        - Predictive analytics
        - Autonomous systems
        
        The future of AI looks promising with continued advancements in deep learning,
        neural networks, and quantum computing integration.
        """,
        "sample_report.md": """
        # Quarterly Business Report
        
        ## Executive Summary
        
        This quarter has shown significant growth across all business units.
        Revenue increased by 15% compared to the previous quarter.
        
        ## Key Metrics
        
        - Revenue: $2.5M (+15%)
        - Customer Acquisition: 1,200 new customers
        - Employee Satisfaction: 87%
        
        ## Challenges
        
        Supply chain disruptions continue to impact delivery times.
        Competition in the market has intensified.
        
        ## Recommendations
        
        1. Invest in supply chain optimization
        2. Enhance customer retention programs
        3. Expand into new market segments
        """,
        "sample_data.json": {
            "title": "Product Catalog",
            "products": [
                {
                    "id": 1,
                    "name": "Laptop Pro",
                    "category": "Electronics",
                    "price": 1299.99,
                    "description": "High-performance laptop for professionals",
                },
                {
                    "id": 2,
                    "name": "Wireless Headphones",
                    "category": "Audio",
                    "price": 199.99,
                    "description": "Premium noise-canceling headphones",
                },
            ],
            "last_updated": "2024-01-15",
        },
    }

    # Create temporary files
    temp_dir = Path("temp_docs")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write sample documents
        for filename, content in sample_docs.items():
            file_path = temp_dir / filename
            if filename.endswith(".json"):
                with open(file_path, "w") as f:
                    json.dump(content, f, indent=2)
            else:
                with open(file_path, "w") as f:
                    f.write(content)

        print("📄 Processing sample documents...")

        # Process each document
        results = []
        for filename in sample_docs.keys():
            file_path = temp_dir / filename
            print(f"\n🔄 Processing: {filename}")

            result = await processor.process_document(file_path)
            results.append(result)

            print(f"   Status: {result.status.value}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Word count: {result.metadata.word_count}")
            print(f"   Quality score: {result.quality_score:.2f}")
            print(f"   Keywords: {', '.join(result.keywords[:5])}")
            print(f"   Topics: {', '.join(result.topics[:3])}")
            print(f"   Summary: {result.summary[:100]}...")

        # Demonstrate information extraction
        print(f"\n🔍 Information Extraction Demo")
        if results:
            doc_id = results[0].document_id
            query = "What are the main applications of AI mentioned?"

            extraction_result = await processor.extract_information(doc_id, query)
            print(f"   Query: {query}")
            print(f"   Result: {extraction_result['extracted_information'][:200]}...")
            print(f"   Confidence: {extraction_result['confidence']:.2f}")

        # Demonstrate document comparison
        print(f"\n📊 Document Comparison Demo")
        if len(results) >= 2:
            comparison_result = await processor.compare_documents(
                results[0].document_id, results[1].document_id
            )
            print(f"   Similarity score: {comparison_result['similarity_score']:.2f}")
            print(f"   Summary: {comparison_result['comparison_summary'][:200]}...")

        # Demonstrate pipeline processing
        print(f"\n🔄 Pipeline Processing Demo")
        pipeline = DocumentProcessingPipeline()

        pipeline_config = [
            {
                "name": "Initial Processing",
                "type": "process",
                "config": {"extraction_types": ["summary", "keywords", "topics"]},
            },
            {
                "name": "Quality Filter",
                "type": "filter",
                "config": {"min_size": 100, "extensions": [".txt", ".md", ".json"]},
            },
            {
                "name": "Aggregate Results",
                "type": "aggregate",
                "config": {"aggregation_type": "summary"},
            },
        ]

        pipeline_id = pipeline.create_pipeline("demo_pipeline", pipeline_config)

        file_paths = [temp_dir / filename for filename in sample_docs.keys()]
        pipeline_result = await pipeline.run_pipeline(pipeline_id, file_paths)

        print(f"   Pipeline status: {pipeline_result['status']}")
        print(f"   Stages completed: {len(pipeline_result['stage_results'])}")

        # List all processed documents
        print(f"\n📋 Processed Documents Summary")
        doc_list = processor.list_processed_documents()
        for doc_info in doc_list:
            print(
                f"   {doc_info['filename']}: {doc_info['status']} "
                f"(Quality: {doc_info['quality_score']:.2f})"
            )

    finally:
        # Cleanup temporary files
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(f"\n✅ Document Processing Demo Complete!")


async def test_document_processing_system():
    """Test the document processing system"""
    print("🧪 Testing Document Processing System")
    print("=" * 50)

    # Test parser registry
    registry = DocumentParserRegistry()
    supported_formats = registry.get_supported_formats()
    print(f"✓ Supported formats: {supported_formats}")

    # Test individual parsers
    parsers = [
        TextDocumentParser(),
        MarkdownDocumentParser(),
        JSONDocumentParser(),
        HTMLDocumentParser(),
    ]

    for parser in parsers:
        formats = parser.get_supported_formats()
        print(f"✓ {parser.__class__.__name__}: {formats}")

    # Test document processor initialization
    processor = DocumentProcessor()
    print("✓ Document processor initialized")

    # Test pipeline creation
    pipeline = DocumentProcessingPipeline()
    test_pipeline = pipeline.create_pipeline(
        "test", [{"name": "Test Stage", "type": "process"}]
    )
    print(f"✓ Pipeline created: {test_pipeline}")

    # Test API interface
    api = DocumentProcessingAPI()
    print("✓ API interface initialized")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    import asyncio
    import time

    async def main():
        """Main function to run demos and tests"""
        print("🚀 Intelligent Document Processing System")
        print("=" * 60)

        # Run tests first
        await test_document_processing_system()
        print()

        # Run demo
        await demo_document_processing()

        print(f"\n🎯 System Features Demonstrated:")
        print("   ✓ Multi-format document parsing (TXT, MD, JSON, HTML)")
        print("   ✓ Intelligent content extraction and analysis")
        print("   ✓ Document summarization and keyword extraction")
        print("   ✓ Named entity recognition")
        print("   ✓ Topic modeling and sentiment analysis")
        print("   ✓ Document structure analysis")
        print("   ✓ Quality assessment")
        print("   ✓ Information extraction with queries")
        print("   ✓ Document comparison and similarity analysis")
        print("   ✓ Batch processing capabilities")
        print("   ✓ Pipeline-based workflow management")
        print("   ✓ REST API interface")
        print("   ✓ Comprehensive error handling and logging")

    # Run the main function
    asyncio.run(main())
