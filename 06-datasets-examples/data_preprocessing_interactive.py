# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore marimo dspy

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from inspect import cleandoc
    from pathlib import Path

    import marimo as mo
    from data_preprocessing import DataPreprocessor, DataValidator, PreprocessingConfig
    from dataset_management import DatasetManager
    from dspy import Example
    from marimo import output

    # Add current directory to path for imports
    sys.path.append(str(Path(__file__).parent))

    # Initialize components
    dataset_manager = DatasetManager()
    data_validator = DataValidator()

    return (
        DataPreprocessor,
        PreprocessingConfig,
        cleandoc,
        data_validator,
        dataset_manager,
        mo,
        output,
    )


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # 🧹 Interactive Data Preprocessing

            Welcome to the hands-on data preprocessing workshop! Here you'll learn to:

            - Configure preprocessing pipelines
            - Clean and normalize text data
            - Apply quality filters
            - Validate preprocessing results
            - Compare before/after quality metrics

            Let's clean up some data! ✨
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(mo, output):
    # Data Loading Section
    cell2_desc = mo.md("## 📁 Step 1: Load Your Dataset")

    dataset_selector = mo.ui.dropdown(
        options=["QA Dataset", "Classification Dataset", "RAG Dataset"],
        value="QA Dataset",
        label="Choose dataset to preprocess",
    )

    load_button = mo.ui.run_button(label="Load Dataset", kind="info")

    cell2_content = mo.vstack([dataset_selector, load_button])

    cell2_out = mo.vstack([cell2_desc, cell2_content])
    output.replace(cell2_out)
    return dataset_selector, load_button


@app.cell
def _(dataset_manager, dataset_selector, load_button):
    # Load selected dataset
    examples = []
    dataset_info = {}

    if load_button.value:
        try:
            if dataset_selector.value == "QA Dataset":
                examples = dataset_manager.load_from_json(
                    "06-datasets-examples/data/sample_qa.json"
                )
                _text_fields = ["question", "answer"]
            elif dataset_selector.value == "Classification Dataset":
                examples = dataset_manager.load_from_csv(
                    "06-datasets-examples/data/sample_classification.csv",
                    "text",
                    "label",
                )
                _text_fields = ["text"]
            else:  # RAG Dataset
                examples = dataset_manager.load_from_jsonl(
                    "06-datasets-examples/data/sample_rag.jsonl"
                )
                _text_fields = ["question", "context", "answer"]

            dataset_info = {
                "name": dataset_selector.value,
                "examples": len(examples),
                "text_fields": _text_fields,
                "success": True,
            }
        except Exception as e:
            dataset_info = {
                "name": dataset_selector.value,
                "examples": 0,
                "text_fields": [],
                "success": False,
                "error": str(e),
            }

    return dataset_info, examples


@app.cell
def _(cleandoc, dataset_info, mo, output):
    # Display loading results
    if dataset_info:
        if dataset_info["success"]:
            cell4_out = mo.md(
                cleandoc(
                    f"""
                    ✅ **Loaded {dataset_info['examples']} examples**
                    - Dataset: {dataset_info['name']}
                    - Text fields: {', '.join(dataset_info['text_fields'])}
                    """
                )
            )
        else:
            cell4_out = mo.md(
                cleandoc(
                    f"""
                    ❌ **Loading failed**
                    - Dataset: {dataset_info['name']}
                    - Error: {dataset_info.get('error', 'Unknown error')}
                    """
                )
            )
    else:
        cell4_out = mo.md("*Click the load button above to load data*")

    output.replace(cell4_out)
    return


@app.cell
def _(examples, mo, output):
    # Preprocessing Configuration
    if not examples:
        cell5_desc = mo.md("*Load data first to configure preprocessing*")
        cell5_content = mo.md("")
        preprocessing_controls = None
    else:
        cell5_desc = mo.md("## ⚙️ Step 2: Configure Preprocessing")

        # Text cleaning options
        cell5_cleaning_desc = mo.md("### Text Cleaning Options")

        cell5_remove_whitespace = mo.ui.checkbox(
            value=True, label="Remove extra whitespace"
        )

        cell5_normalize_unicode = mo.ui.checkbox(
            value=True, label="Normalize Unicode characters"
        )

        cell5_remove_html = mo.ui.checkbox(value=True, label="Remove HTML tags")

        cell5_normalize_quotes = mo.ui.checkbox(
            value=True, label="Normalize quotes and dashes"
        )

        cell5_lowercase = mo.ui.checkbox(value=False, label="Convert to lowercase")

        # Quality filters
        cell5_filters_desc = mo.md("### Quality Filters")

        cell5_min_length = mo.ui.slider(
            start=0,
            stop=100,
            step=5,
            value=10,
            label="Minimum text length (characters)",
        )

        cell5_min_words = mo.ui.slider(
            start=0, stop=20, step=1, value=3, label="Minimum word count"
        )

        cell5_remove_duplicates = mo.ui.checkbox(
            value=True, label="Remove duplicate examples"
        )

        preprocessing_controls = {
            "remove_whitespace": cell5_remove_whitespace,
            "normalize_unicode": cell5_normalize_unicode,
            "remove_html": cell5_remove_html,
            "normalize_quotes": cell5_normalize_quotes,
            "lowercase": cell5_lowercase,
            "min_length": cell5_min_length,
            "min_words": cell5_min_words,
            "remove_duplicates": cell5_remove_duplicates,
        }

        cell5_content = mo.vstack(
            [
                cell5_cleaning_desc,
                mo.hstack(
                    [
                        cell5_remove_whitespace,
                        cell5_normalize_unicode,
                        cell5_remove_html,
                    ]
                ),
                mo.hstack(
                    [cell5_normalize_quotes, cell5_lowercase, cell5_remove_duplicates]
                ),
                cell5_filters_desc,
                cell5_min_length,
                cell5_min_words,
            ]
        )

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (preprocessing_controls,)


@app.cell
def _(PreprocessingConfig, mo, output, preprocessing_controls):
    # Create preprocessing config from UI controls
    config = None
    if preprocessing_controls:
        config = PreprocessingConfig(
            remove_extra_whitespace=preprocessing_controls["remove_whitespace"].value,
            normalize_unicode=preprocessing_controls["normalize_unicode"].value,
            remove_html_tags=preprocessing_controls["remove_html"].value,
            normalize_quotes=preprocessing_controls["normalize_quotes"].value,
            normalize_dashes=preprocessing_controls["normalize_quotes"].value,
            lowercase=preprocessing_controls["lowercase"].value,
            min_text_length=preprocessing_controls["min_length"].value,
            min_word_count=preprocessing_controls["min_words"].value,
            remove_duplicates=preprocessing_controls["remove_duplicates"].value,
        )
        cell6_desc = mo.md("## 🔄 Step 3: Apply Preprocessing")

        preprocess_button = mo.ui.run_button(
            label="Apply Preprocessing", kind="success"
        )
        cell6_content = preprocess_button
    else:
        cell6_desc = mo.md("*Configure preprocessing settings above*")
        cell6_content = mo.md("")
        preprocess_button = None

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return config, preprocess_button


@app.cell
def _(DataPreprocessor, config, dataset_info, examples, preprocess_button):
    # Execute Preprocessing
    processed_examples = []
    processing_stats = {}

    if (
        preprocess_button is not None
        and preprocess_button.value
        and examples
        and config
    ):
        try:
            _preprocessor = DataPreprocessor(config)
            processed_examples = _preprocessor.preprocess_dataset(
                examples, dataset_info["text_fields"]
            )

            processing_stats = {
                "original_count": len(examples),
                "processed_count": len(processed_examples),
                "filtered_count": len(examples) - len(processed_examples),
                "success": True,
            }
        except Exception as e:
            processing_stats = {
                "original_count": len(examples),
                "processed_count": 0,
                "filtered_count": 0,
                "success": False,
                "error": str(e),
            }

    return processed_examples, processing_stats


@app.cell
def _(cleandoc, mo, output, processing_stats):
    # Display processing results
    if processing_stats:
        if processing_stats["success"]:
            _filter_rate = (
                processing_stats["filtered_count"]
                / processing_stats["original_count"]
                * 100
            )

            cell7_out = mo.md(
                cleandoc(
                    f"""
                    ✅ **Preprocessing Complete**
                    - Original examples: {processing_stats['original_count']}
                    - Processed examples: {processing_stats['processed_count']}
                    - Filtered out: {processing_stats['filtered_count']} ({_filter_rate:.1f}%)
                    """
                )
            )
        else:
            cell7_out = mo.md(
                cleandoc(
                    f"""
                    ❌ **Preprocessing failed**
                    - Error: {processing_stats.get('error', 'Unknown error')}
                    """
                )
            )
    else:
        cell7_out = mo.md("*Click 'Apply Preprocessing' to see results*")

    output.replace(cell7_out)
    return


@app.cell
def _(dataset_info, examples, mo, output, processed_examples):
    # Before/After Comparison Setup
    if not examples or not processed_examples:
        cell8_desc = mo.md("*Process data first to see comparison*")
        cell8_content = mo.md("")
    else:
        cell8_desc = mo.md("## 📊 Step 4: Before/After Comparison")

        example_selector = mo.ui.slider(
            start=0,
            stop=min(len(examples), len(processed_examples)) - 1,
            step=1,
            value=0,
            label="Select example to compare",
        )

        field_selector = mo.ui.dropdown(
            options=dataset_info["text_fields"],
            value=dataset_info["text_fields"][0],
            label="Select field to compare",
        )
        cell8_content = mo.vstack([example_selector, field_selector])

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return example_selector, field_selector


@app.cell
def _(
    cleandoc,
    example_selector,
    examples,
    field_selector,
    mo,
    output,
    processed_examples,
):
    # Display comparison
    if (
        example_selector is None
        or field_selector is None
        or not examples
        or not processed_examples
    ):
        cell9_out = mo.md("*Configure comparison settings above*")

    if example_selector.value < len(examples) and example_selector.value < len(
        processed_examples
    ):

        _original = examples[example_selector.value]
        _processed = processed_examples[example_selector.value]

        _original_text = getattr(_original, field_selector.value, "N/A")
        _processed_text = getattr(_processed, field_selector.value, "N/A")

        _comparison_html = cleandoc(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px;">
                    <h4>🔴 Before Preprocessing</h4>
                    <p><strong>Length:</strong> {len(str(_original_text))} characters</p>
                    <p><strong>Words:</strong> {len(str(_original_text).split())} words</p>
                    <div style="background: white; padding: 10px; border-radius: 4px; margin-top: 10px;">
                        <code>{str(_original_text)[:200]}{'...' if len(str(_original_text)) > 200 else ''}</code>
                    </div>
                </div>
                <div style="background: #d4edda; padding: 15px; border-radius: 8px;">
                    <h4>🟢 After Preprocessing</h4>
                    <p><strong>Length:</strong> {len(str(_processed_text))} characters</p>
                    <p><strong>Words:</strong> {len(str(_processed_text).split())} words</p>
                    <div style="background: white; padding: 10px; border-radius: 4px; margin-top: 10px;">
                        <code>{str(_processed_text)[:200]}{'...' if len(str(_processed_text)) > 200 else ''}</code>
                    </div>
                </div>
            </div>
            """
        )

        cell9_out = mo.Html(_comparison_html)
    else:
        cell9_out = mo.md("*Invalid example index*")

    output.replace(cell9_out)
    return


@app.cell
def _(examples, mo, output, processed_examples):
    # Quality Assessment Setup
    if not examples or not processed_examples:
        cell10_desc = mo.md("*Process data first to assess quality*")
        cell10_content = mo.md("")
        assess_button = None
    else:
        cell10_desc = mo.md("## 📈 Step 5: Quality Assessment")

        assess_button = mo.ui.button(label="Assess Quality", kind="info")
        cell10_content = assess_button

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return (assess_button,)


@app.cell
def _(
    assess_button,
    data_validator,
    dataset_info,
    examples,
    processed_examples,
):
    # Execute Quality Assessment
    quality_comparison = {}

    if (
        assess_button is not None
        and assess_button.value
        and examples
        and processed_examples
    ):
        try:
            # Assess original quality
            _original_quality = data_validator.validate_dataset_quality(
                examples, dataset_info["text_fields"]
            )

            # Assess processed quality
            _processed_quality = data_validator.validate_dataset_quality(
                processed_examples, dataset_info["text_fields"]
            )

            quality_comparison = {
                "original": _original_quality,
                "processed": _processed_quality,
                "success": True,
            }
        except Exception as e:
            quality_comparison = {"success": False, "error": str(e)}

    return (quality_comparison,)


@app.cell
def _(cleandoc, mo, output, quality_comparison):
    # Display quality results
    if not quality_comparison:
        cell11_out = mo.md("*Click 'Assess Quality' to see results*")
    else:
        if quality_comparison["success"]:
            _original = quality_comparison["original"]
            _processed = quality_comparison["processed"]

            _improvement = (
                _processed["overall_quality_score"] - _original["overall_quality_score"]
            )

            _quality_html = cleandoc(
                f"""
                <div style="background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3>📊 Quality Improvement Analysis</h3>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0;">
                        <div style="text-align: center;">
                            <h4>Original Quality</h4>
                            <div style="font-size: 24px; color: #dc3545;">
                                {_original['overall_quality_score']:.1%}
                            </div>
                        </div>
                        <div style="text-align: center;">
                            <h4>Processed Quality</h4>
                            <div style="font-size: 24px; color: #28a745;">
                                {_processed['overall_quality_score']:.1%}
                            </div>
                        </div>
                        <div style="text-align: center;">
                            <h4>Improvement</h4>
                            <div style="font-size: 24px; color: {'#28a745' if _improvement > 0 else '#dc3545'};">
                                {_improvement:+.1%}
                            </div>
                        </div>
                    </div>

                    <h4>Detailed Metrics</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Original</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Processed</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Change</th>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;">Valid Examples</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{_original['valid_examples']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{_processed['valid_examples']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{_processed['valid_examples'] - _original['valid_examples']:+d}</td>
                        </tr>
                    </table>
                </div>
                """
            )

            cell11_out = mo.Html(_quality_html)
        else:
            cell11_out = mo.md(
                f"❌ **Quality assessment failed:** {quality_comparison.get('error', 'Unknown error')}"
            )

    output.replace(cell11_out)
    return


@app.cell
def _(mo, output, processed_examples):
    # Export Processed Data Setup
    if not processed_examples:
        cell12_desc = mo.md("*Process data first to export*")
        cell12_content = mo.md("")
        export_format = None
        filename = None
        export_button = None
    else:
        cell12_desc = mo.md("## 💾 Step 6: Export Processed Data")

        export_format = mo.ui.dropdown(
            options=["JSON", "JSONL"], value="JSON", label="Export format"
        )

        filename = mo.ui.text(
            value="processed_dataset", label="Filename (without extension)"
        )

        export_button = mo.ui.button(label="Export Processed Data", kind="success")

        cell12_content = mo.vstack(
            [mo.hstack([export_format, filename]), export_button]
        )

    cell12_out = mo.vstack([cell12_desc, cell12_content])
    output.replace(cell12_out)
    return export_button, export_format, filename


@app.cell
def _(
    dataset_manager,
    export_button,
    export_format,
    filename,
    mo,
    output,
    processed_examples,
):
    # Handle export
    if not processed_examples or not export_button is None or not filename or not export_format:
        cell13_out = mo.md("*Configure export settings above*")
    else:
        if export_button.value and filename.value:
            _export_path = f"{filename.value}.{export_format.value.lower()}"

            try:
                if export_format.value == "JSON":
                    _success = dataset_manager.save_to_json(
                        processed_examples, _export_path
                    )
                else:
                    _success = dataset_manager.save_to_jsonl(
                        processed_examples, _export_path
                    )

                if _success:
                    cell13_out = mo.md(
                        f"✅ **Successfully exported {len(processed_examples)} processed examples to {_export_path}**"
                    )
                else:
                    cell13_out = mo.md("❌ **Export failed**")

            except Exception as e:
                cell13_out = mo.md(f"❌ **Export error:** {str(e)}")
        else:
            cell13_out = mo.md("*Enter filename and click 'Export Processed Data'*")

    output.replace(cell13_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with tips and next steps
    _content = cleandoc(
        """
        ## 🎯 Preprocessing Best Practices

        ### ✅ What You've Learned

        - **Text Cleaning**: Remove noise and standardize formats
        - **Quality Filtering**: Remove low-quality examples
        - **Validation**: Always check quality before and after
        - **Configuration**: Adjust settings based on your data type

        ### 💡 Pro Tips

        1. **Start Conservative**: Begin with minimal preprocessing and gradually add more
        2. **Domain-Specific**: Adjust settings based on your data domain (technical, casual, etc.)
        3. **Quality vs Quantity**: Sometimes filtering improves model performance more than having more data
        4. **Iterative Process**: Preprocessing is often iterative - refine based on results

        ### 🚀 Next Steps

        - **Quality Assessment**: Dive deeper into data quality metrics
        - **Data Exploration**: Explore patterns in your processed data
        - **DSPy Integration**: Use your clean data in DSPy modules

        **Ready to explore more?** Try the quality assessment or data exploration notebooks!
        """
    )

    cell14_out = mo.md(_content)
    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
