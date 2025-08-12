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
    from marimo import output

    # Add current directory to path for imports
    sys.path.append(str(Path(__file__).parent))

    from dataset_management import DatasetManager

    # Initialize dataset manager
    dataset_manager = DatasetManager()
    return cleandoc, dataset_manager, mo, output


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # 📚 Dataset Loading Basics

            Welcome to the interactive guide for loading and working with DSPy datasets!
            In this notebook, you'll learn how to:

            - Load datasets from JSON, JSONL, and CSV formats
            - Create DSPy Example objects
            - Validate and explore loaded data
            - Handle common loading issues

            Let's get started! 🚀
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(mo, output):
    # File Format Selection
    cell2_desc = mo.md("## 📁 Step 1: Choose Your Data Format")

    format_selector = mo.ui.dropdown(
        options=["JSON", "JSONL", "CSV"],
        value="JSON",
        label="Select data format to explore",
    )

    sample_files = {
        "JSON": "06-datasets-examples/data/sample_qa.json",
        "JSONL": "06-datasets-examples/data/sample_rag.jsonl",
        "CSV": "06-datasets-examples/data/sample_classification.csv",
    }

    cell2_content = format_selector

    cell2_out = mo.vstack([cell2_desc, cell2_content])
    output.replace(cell2_out)
    return format_selector, sample_files


@app.cell
def _(format_selector, mo, output):
    # Interactive Data Loading
    cell3_desc = mo.md("## 🔄 Step 2: Load Your Data")

    load_button = mo.ui.run_button(
        label=f"Load {format_selector.value} Data", kind="success"
    )

    cell3_content = load_button

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (load_button,)


@app.cell
def _(dataset_manager, format_selector, load_button, sample_files):
    # Load data based on selection
    examples = []
    loading_info = {}

    if load_button.value:
        _selected_format = format_selector.value
        _file_path = sample_files[_selected_format]

        try:
            if _selected_format == "JSON":
                examples = dataset_manager.load_from_json(_file_path)
            elif _selected_format == "JSONL":
                examples = dataset_manager.load_from_jsonl(_file_path)
            elif _selected_format == "CSV":
                examples = dataset_manager.load_from_csv(_file_path, "text", "label")

            loading_info = {
                "format": _selected_format,
                "file_path": _file_path,
                "examples_loaded": len(examples),
                "success": True,
            }

        except Exception as e:
            loading_info = {
                "format": _selected_format,
                "file_path": _file_path,
                "examples_loaded": 0,
                "success": False,
                "error": str(e),
            }

    return examples, loading_info


@app.cell
def _(cleandoc, loading_info, mo, output):
    # Display loading results
    if loading_info:
        if loading_info["success"]:
            cell5_out = mo.md(
                cleandoc(
                    f"""
                    ✅ **Successfully loaded {loading_info['examples_loaded']} examples**
                    - Format: {loading_info['format']}
                    - File: {loading_info['file_path']}
                    """
                )
            )
        else:
            cell5_out = mo.md(
                cleandoc(
                    f"""
                    ❌ **Loading failed**
                    - Format: {loading_info['format']}
                    - File: {loading_info['file_path']}
                    - Error: {loading_info.get('error', 'Unknown error')}
                    """
                )
            )
    else:
        cell5_out = mo.md("*Click the load button above to load data*")

    output.replace(cell5_out)
    return


@app.cell
def _(examples, mo, output):
    # Data Exploration Setup
    if not examples:
        cell6_out = mo.md("*Load data first to explore examples*")
        example_index = None
    else:
        cell6_desc = mo.md("## 🔍 Step 3: Explore Your Data")

        # Example selector
        example_index = mo.ui.slider(
            start=1,
            stop=len(examples),
            step=1,
            value=2,
            label=f"Select example (1 to {len(examples)})",
        )

        cell6_content = example_index
        cell6_out = mo.vstack([cell6_desc, cell6_content])

    output.replace(cell6_out)
    return (example_index,)


@app.cell
def _(cleandoc, example_index, examples, mo, output):
    # Display selected example
    if not examples or example_index is None:
        cell7_out = mo.md("*No data available to display*")
    else:
        if example_index.value <= len(examples):
            _selected_example = examples[example_index.value - 1]

            # DSPy Examples can be accessed in multiple ways
            _fields = {}

            # Method 1: Direct attribute access for known fields
            known_fields = [
                "question",
                "answer",
                "text",
                "label",
                "input",
                "output",
                "context",
                "query",
                "response",
                "category",
                "difficulty",
            ]

            for field in known_fields:
                try:
                    if hasattr(_selected_example, field):
                        value = getattr(_selected_example, field)
                        if value is not None:
                            _fields[field] = value
                except Exception as _:
                    continue

            # Method 2: Try __dict__ access
            if not _fields:
                try:
                    for key, value in _selected_example.__dict__.items():
                        if not key.startswith("_") and value is not None:
                            _fields[key] = value
                except Exception as _:
                    pass

            # Method 3: Try accessing as dict-like object
            if not _fields:
                try:
                    # Some DSPy Examples support dict-like access
                    if hasattr(_selected_example, "keys"):
                        for key in _selected_example.keys():
                            _fields[key] = _selected_example[key]
                except Exception as _:
                    pass

            # Create a nice display
            _example_html = cleandoc(
                f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
                    <h4>Example {example_index.value}</h4>
                """
            )

            if _fields:
                for field, value in _fields.items():
                    if isinstance(value, str) and len(value) > 100:
                        _display_value = value[:100] + "..."
                    else:
                        _display_value = str(value)

                    _example_html += cleandoc(
                        f"""
                        <p><strong>{field}:</strong> {_display_value}</p>
                        """
                    )
            else:
                # Fallback: show debug information
                _example_html += cleandoc(
                    f"""
                    <p><strong>Type:</strong> {type(_selected_example).__name__}</p>
                    <p><strong>String representation:</strong> {str(_selected_example)}</p>
                    <p><strong>Dir (non-private):</strong> {[attr for attr in dir(_selected_example) if not attr.startswith('_')]}</p>
                    """
                )

            _example_html += "</div>"

            cell7_out = mo.Html(_example_html)
        else:
            cell7_out = mo.md("*Invalid example index*")

    output.replace(cell7_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with next steps
    cell8_out = mo.md(
        cleandoc(
            """
            ## 🎯 What's Next?

            Now that you've mastered dataset loading basics, you can:

            1. **Explore Data Preprocessing** - Clean and transform your data
            2. **Assess Data Quality** - Analyze and improve dataset quality
            3. **Interactive Data Exploration** - Dive deeper into your data patterns

            ### 💡 Key Takeaways

            - DSPy Examples are flexible containers for structured data
            - Always validate your data after loading
            - Different formats serve different purposes (JSON for structure, JSONL for streaming, CSV for tabular)
            - Interactive exploration helps you understand your data better

            **Ready for the next challenge?** Try the data preprocessing notebook! 🚀
            """
        )
    )

    output.replace(cell8_out)
    return


if __name__ == "__main__":
    app.run()
