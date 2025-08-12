"""
Interactive Dataset Explorer for DSPy Examples

This Marimo notebook provides an interactive interface for exploring and analyzing
DSPy datasets. It includes visualization tools, statistical analysis, and data
quality assessment capabilities.

Features:
- Interactive data loading and exploration
- Statistical visualizations and analysis
- Data quality assessment dashboard
- Field-specific analysis and filtering
- Export capabilities for processed data
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import DataPreprocessor, DataValidator, PreprocessingConfig

# Import our custom modules
from dataset_management import DatasetManager, DatasetStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset_explorer():
    """Create the main dataset explorer interface"""

    # Title and introduction
    mo.md(
        """
    # 📊 DSPy Dataset Explorer

    Interactive tool for exploring and analyzing DSPy datasets. Load your data,
    explore statistics, assess quality, and visualize patterns.
    """
    )

    # File upload and dataset selection
    file_selector = mo.ui.file(
        filetypes=[".json", ".jsonl", ".csv"],
        multiple=False,
        label="Upload Dataset File",
    )

    # Sample dataset selector
    sample_datasets = {
        "Sample QA Dataset": "data/sample_qa.json",
        "Sample Classification": "data/sample_classification.csv",
        "Sample RAG Dataset": "data/sample_rag.jsonl",
    }

    sample_selector = mo.ui.dropdown(
        options=list(sample_datasets.keys()),
        value="Sample QA Dataset",
        label="Or select a sample dataset",
    )

    # Display file selection interface
    mo.md("## 📁 Dataset Selection")
    mo.hstack([file_selector, sample_selector])

    return file_selector, sample_selector, sample_datasets


def load_selected_dataset(file_selector, sample_selector, sample_datasets):
    """Load the selected dataset"""

    manager = DatasetManager()
    examples = []
    dataset_info = {}

    try:
        if file_selector.value:
            # Load uploaded file
            file_path = file_selector.value[0].name
            file_extension = Path(file_path).suffix.lower()

            if file_extension == ".json":
                examples = manager.load_from_json(file_path)
            elif file_extension == ".jsonl":
                examples = manager.load_from_jsonl(file_path)
            elif file_extension == ".csv":
                # For CSV, assume text and label columns
                examples = manager.load_from_csv(file_path, "text", "label")

            dataset_info = {
                "name": file_path,
                "source": "uploaded",
                "format": file_extension,
            }

        else:
            # Load sample dataset
            dataset_name = sample_selector.value
            file_path = sample_datasets[dataset_name]
            file_extension = Path(file_path).suffix.lower()

            if file_extension == ".json":
                examples = manager.load_from_json(file_path)
            elif file_extension == ".jsonl":
                examples = manager.load_from_jsonl(file_path)
            elif file_extension == ".csv":
                examples = manager.load_from_csv(file_path, "text", "label")

            dataset_info = {
                "name": dataset_name,
                "source": "sample",
                "format": file_extension,
                "path": file_path,
            }

    except Exception as e:
        mo.md(f"❌ **Error loading dataset:** {str(e)}")
        return [], {}

    if examples:
        mo.md(f"✅ **Loaded {len(examples)} examples** from {dataset_info['name']}")
    else:
        mo.md("⚠️ **No examples loaded.** Please check your file format.")

    return examples, dataset_info


def create_dataset_overview(examples, dataset_info):
    """Create dataset overview section"""

    if not examples:
        return

    mo.md("## 📋 Dataset Overview")

    # Get dataset statistics
    manager = DatasetManager()
    stats = manager.get_dataset_stats(examples)

    # Create overview metrics
    col1, col2, col3, col4 = mo.ui.tabs(
        {
            "Basic Info": create_basic_info_tab(stats, dataset_info),
            "Field Analysis": create_field_analysis_tab(stats),
            "Text Statistics": create_text_stats_tab(stats),
            "Data Quality": create_quality_tab(examples, stats),
        }
    )

    return col1, col2, col3, col4


def create_basic_info_tab(stats: DatasetStats, dataset_info: Dict):
    """Create basic information tab"""

    info_html = f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h4>📊 Dataset Metrics</h4>
            <p><strong>Total Examples:</strong> {stats.total_examples:,}</p>
            <p><strong>Fields:</strong> {len(stats.field_counts)}</p>
            <p><strong>Format:</strong> {dataset_info.get('format', 'Unknown')}</p>
        </div>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h4>📁 Source Information</h4>
            <p><strong>Name:</strong> {dataset_info.get('name', 'Unknown')}</p>
            <p><strong>Source:</strong> {dataset_info.get('source', 'Unknown')}</p>
            <p><strong>Path:</strong> {dataset_info.get('path', 'N/A')}</p>
        </div>
    </div>
    """

    return mo.Html(info_html)


def create_field_analysis_tab(stats: DatasetStats):
    """Create field analysis tab"""

    if not stats.field_counts:
        return mo.md("No field data available.")

    # Create field completeness chart
    fields = list(stats.field_counts.keys())
    completeness = [
        stats.field_counts[field] / stats.total_examples * 100 for field in fields
    ]

    fig = px.bar(
        x=fields,
        y=completeness,
        title="Field Completeness (%)",
        labels={"x": "Fields", "y": "Completeness (%)"},
        color=completeness,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=400)

    # Field types table
    field_data = []
    for field in fields:
        field_data.append(
            {
                "Field": field,
                "Type": stats.field_types.get(field, "Unknown"),
                "Count": stats.field_counts[field],
                "Missing": stats.missing_values.get(field, 0),
                "Unique": stats.unique_values.get(field, 0),
                "Completeness": f"{completeness[fields.index(field)]:.1f}%",
            }
        )

    df = pd.DataFrame(field_data)

    return mo.vstack(
        [mo.Html(fig.to_html()), mo.md("### Field Details"), mo.ui.table(df)]
    )


def create_text_stats_tab(stats: DatasetStats):
    """Create text statistics tab"""

    text_fields = [
        field for field, avg_len in stats.avg_text_length.items() if avg_len > 0
    ]

    if not text_fields:
        return mo.md("No text fields found in the dataset.")

    # Create text length comparison chart
    field_lengths = [stats.avg_text_length[field] for field in text_fields]

    fig = px.bar(
        x=text_fields,
        y=field_lengths,
        title="Average Text Length by Field",
        labels={"x": "Text Fields", "y": "Average Length (characters)"},
        color=field_lengths,
        color_continuous_scale="viridis",
    )
    fig.update_layout(height=400)

    # Text statistics table
    text_data = []
    for field in text_fields:
        text_data.append(
            {
                "Field": field,
                "Avg Length": f"{stats.avg_text_length[field]:.1f}",
                "Unique Values": stats.unique_values.get(field, 0),
                "Completeness": f"{stats.field_counts[field] / stats.total_examples * 100:.1f}%",
            }
        )

    df = pd.DataFrame(text_data)

    return mo.vstack(
        [mo.Html(fig.to_html()), mo.md("### Text Field Statistics"), mo.ui.table(df)]
    )


def create_quality_tab(examples, stats: DatasetStats):
    """Create data quality assessment tab"""

    # Perform quality assessment
    validator = DataValidator()

    # Determine text fields for validation
    text_fields = [
        field for field, avg_len in stats.avg_text_length.items() if avg_len > 0
    ]

    if not text_fields:
        return mo.md("No text fields available for quality assessment.")

    quality_metrics = validator.validate_dataset_quality(examples, text_fields)

    # Overall quality score
    overall_score = quality_metrics["overall_quality_score"]
    score_color = (
        "green" if overall_score > 0.8 else "orange" if overall_score > 0.6 else "red"
    )

    quality_html = f"""
    <div style="text-align: center; margin: 20px 0;">
        <h3>Overall Quality Score</h3>
        <div style="font-size: 48px; color: {score_color}; font-weight: bold;">
            {overall_score:.1%}
        </div>
    </div>
    """

    # Field-specific quality metrics
    field_quality_data = []
    for field, metrics in quality_metrics["field_quality"].items():
        field_quality_data.append(
            {
                "Field": field,
                "Valid Count": metrics["valid_count"],
                "Avg Length": f"{metrics['avg_length']:.1f}",
                "Avg Words": f"{metrics['avg_word_count']:.1f}",
                "Issues": len(metrics["quality_issues"]),
            }
        )

    df = pd.DataFrame(field_quality_data)

    # Quality issues summary
    all_issues = []
    for field, metrics in quality_metrics["field_quality"].items():
        for issue in metrics["quality_issues"]:
            all_issues.append(f"{field}: {issue}")

    issues_html = (
        f"""
    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>⚠️ Quality Issues Found</h4>
        <ul>
            {''.join(f'<li>{issue}</li>' for issue in all_issues[:10])}
            {f'<li>... and {len(all_issues) - 10} more</li>' if len(all_issues) > 10 else ''}
        </ul>
    </div>
    """
        if all_issues
        else """
    <div style="background: #d4edda; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>✅ No Quality Issues Found</h4>
        <p>Your dataset appears to be in good condition!</p>
    </div>
    """
    )

    return mo.vstack(
        [
            mo.Html(quality_html),
            mo.md("### Field Quality Metrics"),
            mo.ui.table(df),
            mo.Html(issues_html),
        ]
    )


def create_data_viewer(examples):
    """Create interactive data viewer"""

    if not examples:
        return mo.md("No data to display.")

    mo.md("## 🔍 Data Viewer")

    # Convert examples to DataFrame for display
    data_rows = []
    for i, example in enumerate(examples[:100]):  # Limit to first 100 for performance
        row = {"Index": i}
        for key, value in example.__dict__.items():
            if not key.startswith("_"):
                # Truncate long text for display
                if isinstance(value, str) and len(value) > 100:
                    row[key] = value[:100] + "..."
                else:
                    row[key] = value
        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Add pagination controls
    page_size = mo.ui.slider(
        start=10, stop=50, step=10, value=20, label="Rows per page"
    )

    # Search functionality
    search_box = mo.ui.text(placeholder="Search in data...", label="Search")

    return mo.vstack(
        [mo.hstack([page_size, search_box]), mo.ui.table(df.head(page_size.value))]
    )


def create_preprocessing_interface(examples):
    """Create preprocessing configuration interface"""

    if not examples:
        return mo.md("No data available for preprocessing.")

    mo.md("## ⚙️ Data Preprocessing")

    # Preprocessing configuration controls
    config_controls = {
        "remove_extra_whitespace": mo.ui.checkbox(
            value=True, label="Remove extra whitespace"
        ),
        "normalize_unicode": mo.ui.checkbox(value=True, label="Normalize Unicode"),
        "remove_html_tags": mo.ui.checkbox(value=True, label="Remove HTML tags"),
        "normalize_quotes": mo.ui.checkbox(value=True, label="Normalize quotes"),
        "lowercase": mo.ui.checkbox(value=False, label="Convert to lowercase"),
        "remove_duplicates": mo.ui.checkbox(value=True, label="Remove duplicates"),
    }

    # Length filters
    min_length = mo.ui.number(
        start=0, stop=1000, step=1, value=10, label="Minimum text length"
    )

    max_length = mo.ui.number(
        start=100, stop=10000, step=100, value=1000, label="Maximum text length"
    )

    # Preprocessing button
    preprocess_button = mo.ui.button(label="Apply Preprocessing", kind="success")

    # Display controls
    controls_grid = mo.Html(
        f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
        <div>{config_controls['remove_extra_whitespace']}</div>
        <div>{config_controls['normalize_unicode']}</div>
        <div>{config_controls['remove_html_tags']}</div>
        <div>{config_controls['normalize_quotes']}</div>
        <div>{config_controls['lowercase']}</div>
        <div>{config_controls['remove_duplicates']}</div>
    </div>
    """
    )

    return mo.vstack(
        [controls_grid, mo.hstack([min_length, max_length]), preprocess_button]
    )


def create_export_interface(examples, dataset_info):
    """Create data export interface"""

    if not examples:
        return mo.md("No data available for export.")

    mo.md("## 💾 Export Data")

    # Export format selector
    export_format = mo.ui.dropdown(
        options=["JSON", "JSONL", "CSV"], value="JSON", label="Export format"
    )

    # Export filename
    default_name = f"processed_{dataset_info.get('name', 'dataset')}"
    filename = mo.ui.text(value=default_name, label="Filename (without extension)")

    # Export button
    export_button = mo.ui.button(label="Export Dataset", kind="success")

    # Export statistics
    export_stats = mo.Html(
        f"""
    <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h4>📊 Export Information</h4>
        <p><strong>Examples to export:</strong> {len(examples):,}</p>
        <p><strong>Estimated file size:</strong> ~{len(str(examples)) / 1024:.1f} KB</p>
    </div>
    """
    )

    return mo.vstack(
        [mo.hstack([export_format, filename]), export_button, export_stats]
    )


# Main application
def main():
    """Main application function"""

    # Create the dataset explorer interface
    file_selector, sample_selector, sample_datasets = create_dataset_explorer()

    # Load selected dataset
    examples, dataset_info = load_selected_dataset(
        file_selector, sample_selector, sample_datasets
    )

    if examples:
        # Create overview tabs
        create_dataset_overview(examples, dataset_info)

        # Create data viewer
        create_data_viewer(examples)

        # Create preprocessing interface
        create_preprocessing_interface(examples)

        # Create export interface
        create_export_interface(examples, dataset_info)

    # Footer
    mo.md(
        """
    ---

    ### 💡 Tips for Dataset Exploration

    1. **Start with the overview** to understand your data structure
    2. **Check data quality** before preprocessing
    3. **Use preprocessing** to clean and standardize your data
    4. **Export processed data** for use in DSPy applications

    ### 🔗 Related Modules

    - **Module 05**: Evaluation & Metrics
    - **Module 07**: Tracing & Debugging
    - **Module 08**: Custom DSPy Modules
    """
    )


if __name__ == "__main__":
    main()
