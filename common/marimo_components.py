"""
Reusable Marimo UI components for DSPy integration.

This module provides interactive UI elements specifically designed for DSPy learning,
including parameter controls, result visualization, and reactive update handlers.
"""

# Standard Library
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-Party Library
import marimo as mo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Local Modules
from .config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Control Components
# =============================================================================


class DSPyParameterPanel:
    """
    A comprehensive parameter control panel for DSPy modules.

    This component provides common parameter controls like temperature,
    max_tokens, model selection, etc., with reactive updates.
    """

    def __init__(
        self,
        show_model_selection: bool = True,
        show_temperature: bool = True,
        show_max_tokens: bool = True,
        show_provider_selection: bool = True,
        custom_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the parameter panel.

        Args:
            show_model_selection: Show model selection dropdown
            show_temperature: Show temperature slider
            show_max_tokens: Show max tokens slider
            show_provider_selection: Show LLM provider selection
            custom_params: Additional custom parameters
        """
        self.config = get_config()
        self.custom_params = custom_params or {}

        # Build parameter controls
        self.controls = {}

        if show_provider_selection:
            available_providers = self.config.get_available_llm_providers()
            if available_providers:
                self.controls["provider"] = mo.ui.dropdown(
                    options=available_providers,
                    value=self.config.default_llm_provider,
                    label="LLM Provider",
                )

        if show_model_selection:
            self.controls["model"] = mo.ui.text(
                value=self.config.default_model,
                label="Model Name",
                placeholder="e.g., gpt-4o-mini, claude-3-sonnet",
            )

        if show_temperature:
            self.controls["temperature"] = mo.ui.slider(
                start=0.0, stop=2.0, step=0.1, value=0.7, label="Temperature"
            )

        if show_max_tokens:
            self.controls["max_tokens"] = mo.ui.slider(
                start=50, stop=4000, step=50, value=1000, label="Max Tokens"
            )

        # Add custom parameters
        for param_name, param_config in self.custom_params.items():
            if param_config["type"] == "slider":
                self.controls[param_name] = mo.ui.slider(
                    start=param_config.get("min", 0),
                    stop=param_config.get("max", 100),
                    step=param_config.get("step", 1),
                    value=param_config.get("default", 50),
                    label=param_config.get("label", param_name),
                )
            elif param_config["type"] == "dropdown":
                self.controls[param_name] = mo.ui.dropdown(
                    options=param_config["options"],
                    value=param_config.get("default"),
                    label=param_config.get("label", param_name),
                )
            elif param_config["type"] == "text":
                self.controls[param_name] = mo.ui.text(
                    value=param_config.get("default", ""),
                    label=param_config.get("label", param_name),
                    placeholder=param_config.get("placeholder", ""),
                )

    def get_values(self) -> Dict[str, Any]:
        """Get current values from all controls."""
        return {name: control.value for name, control in self.controls.items()}

    def render(self) -> mo.Html:
        """Render the parameter panel."""
        if not self.controls:
            return mo.md("No parameters to configure.")

        # Create a grid layout for the controls
        control_elements = []
        for name, control in self.controls.items():
            control_elements.append(
                mo.vstack(
                    [
                        mo.md(
                            f"**{control.label if hasattr(control, 'label') else name}**"
                        ),
                        control,
                    ]
                )
            )

        return mo.vstack(
            [mo.md("## Parameters"), mo.hstack(control_elements, wrap=True)]
        )


class SignatureBuilder:
    """Interactive DSPy signature builder component."""

    def __init__(self):
        """Initialize the signature builder."""
        self.input_fields = mo.ui.array(
            [
                mo.ui.text(placeholder="Field name", label="Input Field"),
            ]
        )

        self.output_fields = mo.ui.array(
            [
                mo.ui.text(placeholder="Field name", label="Output Field"),
            ]
        )

        self.signature_name = mo.ui.text(
            placeholder="MySignature", label="Signature Name"
        )

        self.docstring = mo.ui.text_area(
            placeholder="Describe what this signature does...", label="Description"
        )

    def generate_signature_code(self) -> str:
        """Generate DSPy signature code from current inputs."""
        name = self.signature_name.value or "MySignature"
        doc = self.docstring.value or "Generated signature"

        input_names = [
            field.value for field in self.input_fields.value if field.value.strip()
        ]
        output_names = [
            field.value for field in self.output_fields.value if field.value.strip()
        ]

        code_lines = [f"class {name}(dspy.Signature):", f'    """{doc}"""', ""]

        for input_name in input_names:
            code_lines.append(f"    {input_name} = dspy.InputField()")

        if input_names and output_names:
            code_lines.append("")

        for output_name in output_names:
            code_lines.append(f"    {output_name} = dspy.OutputField()")

        return "\n".join(code_lines)

    def render(self) -> mo.Html:
        """Render the signature builder."""
        return mo.vstack(
            [
                mo.md("## DSPy Signature Builder"),
                mo.hstack(
                    [
                        mo.vstack([self.signature_name, self.docstring]),
                        mo.vstack(
                            [
                                mo.md("**Input Fields**"),
                                self.input_fields,
                                mo.md("**Output Fields**"),
                                self.output_fields,
                            ]
                        ),
                    ]
                ),
                mo.md("### Generated Code"),
                mo.md(f"```python\n{self.generate_signature_code()}\n```"),
            ]
        )


# =============================================================================
# Result Visualization Components
# =============================================================================


class DSPyResultViewer:
    """Component for displaying DSPy prediction results."""

    def __init__(self, result: Any = None):
        """
        Initialize the result viewer.

        Args:
            result: DSPy prediction result to display
        """
        self.result = result

    def render(self) -> mo.Html:
        """Render the result viewer."""
        if self.result is None:
            return mo.md("*No results to display*")

        # Handle different types of results
        if hasattr(self.result, "__dict__"):
            # DSPy Prediction object
            result_dict = {}
            for key, value in self.result.__dict__.items():
                if not key.startswith("_"):
                    result_dict[key] = str(value)

            return mo.vstack(
                [
                    mo.md("## Prediction Results"),
                    mo.ui.table(
                        data=pd.DataFrame([result_dict]).T.reset_index(),
                        columns=["Field", "Value"],
                    ),
                ]
            )

        elif isinstance(self.result, dict):
            return mo.vstack(
                [
                    mo.md("## Results"),
                    mo.ui.table(
                        data=pd.DataFrame([self.result]).T.reset_index(),
                        columns=["Field", "Value"],
                    ),
                ]
            )

        else:
            return mo.vstack(
                [mo.md("## Result"), mo.md(f"```\n{str(self.result)}\n```")]
            )


class OptimizationProgressViewer:
    """Component for visualizing DSPy optimization progress."""

    def __init__(self):
        """Initialize the optimization progress viewer."""
        self.progress_data = []
        self.metrics_data = []

    def add_progress_point(
        self,
        step: int,
        metric_value: float,
        metric_name: str = "Score",
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Add a progress data point.

        Args:
            step: Optimization step number
            metric_value: Primary metric value
            metric_name: Name of the primary metric
            additional_metrics: Additional metrics to track
        """
        point = {"step": step, metric_name: metric_value, "timestamp": datetime.now()}

        if additional_metrics:
            point.update(additional_metrics)

        self.progress_data.append(point)

    def render(self) -> mo.Html:
        """Render the optimization progress."""
        if not self.progress_data:
            return mo.md("*No optimization data to display*")

        df = pd.DataFrame(self.progress_data)

        # Create progress chart
        fig = px.line(
            df,
            x="step",
            y=[col for col in df.columns if col not in ["step", "timestamp"]],
            title="Optimization Progress",
            labels={"value": "Metric Value", "step": "Optimization Step"},
        )

        fig.update_layout(
            xaxis_title="Step", yaxis_title="Metric Value", hovermode="x unified"
        )

        return mo.vstack(
            [
                mo.md("## Optimization Progress"),
                mo.ui.plotly(fig),
                mo.md("### Progress Data"),
                mo.ui.table(df),
            ]
        )


class ComparisonViewer:
    """Component for comparing multiple DSPy results or configurations."""

    def __init__(self):
        """Initialize the comparison viewer."""
        self.comparisons = []

    def add_comparison(
        self,
        name: str,
        result: Any,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a result for comparison.

        Args:
            name: Name/label for this result
            result: The result object
            metrics: Performance metrics
            config: Configuration used
        """
        comparison = {
            "name": name,
            "result": result,
            "metrics": metrics or {},
            "config": config or {},
            "timestamp": datetime.now(),
        }
        self.comparisons.append(comparison)

    def render(self) -> mo.Html:
        """Render the comparison view."""
        if not self.comparisons:
            return mo.md("*No comparisons to display*")

        # Create metrics comparison table
        metrics_data = []
        for comp in self.comparisons:
            row = {"Name": comp["name"]}
            row.update(comp["metrics"])
            metrics_data.append(row)

        metrics_df = pd.DataFrame(metrics_data)

        # Create results display
        result_elements = []
        for comp in self.comparisons:
            result_elements.append(
                mo.vstack(
                    [
                        mo.md(f"### {comp['name']}"),
                        mo.md(f"```\n{str(comp['result'])}\n```"),
                        mo.md(f"*Added: {comp['timestamp'].strftime('%H:%M:%S')}*"),
                    ]
                )
            )

        return mo.vstack(
            [
                mo.md("## Comparison Results"),
                mo.md("### Metrics Comparison"),
                (
                    mo.ui.table(metrics_df)
                    if not metrics_df.empty
                    else mo.md("*No metrics to compare*")
                ),
                mo.md("### Detailed Results"),
                mo.hstack(result_elements, wrap=True),
            ]
        )


# =============================================================================
# Interactive Testing Components
# =============================================================================


class SignatureTester:
    """Interactive component for testing DSPy signatures."""

    def __init__(self, signature_class: Any = None):
        """
        Initialize the signature tester.

        Args:
            signature_class: DSPy signature class to test
        """
        self.signature_class = signature_class
        self.test_inputs = {}
        self.results = []

        # Create input controls based on signature
        if signature_class:
            self._create_input_controls()

    def _create_input_controls(self):
        """Create input controls based on the signature fields."""
        if not self.signature_class:
            return

        # Get input fields from signature
        try:
            sig_instance = self.signature_class()
            for field_name, field_info in sig_instance.model_fields.items():
                if (
                    hasattr(field_info, "json_schema_extra")
                    and field_info.json_schema_extra
                    and field_info.json_schema_extra.get("__dspy_field_type") == "input"
                ):

                    self.test_inputs[field_name] = mo.ui.text_area(
                        placeholder=f"Enter {field_name}...",
                        label=field_name.replace("_", " ").title(),
                    )
        except Exception as e:
            logger.warning(f"Could not create input controls: {e}")

    def add_test_result(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float = 0.0,
    ):
        """
        Add a test result.

        Args:
            inputs: Input values used
            outputs: Output values received
            execution_time: Time taken for execution
        """
        self.results.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "execution_time": execution_time,
                "timestamp": datetime.now(),
            }
        )

    def render(self) -> mo.Html:
        """Render the signature tester."""
        elements = [mo.md("## Signature Tester")]

        if self.signature_class:
            elements.append(mo.md(f"**Testing:** `{self.signature_class.__name__}`"))

        # Input controls
        if self.test_inputs:
            elements.append(mo.md("### Test Inputs"))
            input_elements = [control for control in self.test_inputs.values()]
            elements.append(mo.vstack(input_elements))
        else:
            elements.append(mo.md("*No signature loaded for testing*"))

        # Results display
        if self.results:
            elements.append(mo.md("### Test Results"))

            # Create results table
            results_data = []
            for i, result in enumerate(self.results):
                row = {
                    "Test": i + 1,
                    "Execution Time": f"{result['execution_time']:.3f}s",
                    "Timestamp": result["timestamp"].strftime("%H:%M:%S"),
                }
                # Add input/output summary
                row.update(
                    {
                        f"Input_{k}": (
                            str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                        )
                        for k, v in result["inputs"].items()
                    }
                )
                row.update(
                    {
                        f"Output_{k}": (
                            str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                        )
                        for k, v in result["outputs"].items()
                    }
                )
                results_data.append(row)

            results_df = pd.DataFrame(results_data)
            elements.append(mo.ui.table(results_df))
        else:
            elements.append(mo.md("*No test results yet*"))

        return mo.vstack(elements)


# =============================================================================
# Utility Functions
# =============================================================================


def create_parameter_grid(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create a parameter grid for systematic testing.

    Args:
        parameters: Dictionary mapping parameter names to lists of values

    Returns:
        List of parameter combinations
    """
    import itertools

    param_names = list(parameters.keys())
    param_values = list(parameters.values())

    combinations = []
    for combo in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def format_dspy_result(result: Any) -> str:
    """
    Format a DSPy result for display.

    Args:
        result: DSPy result object

    Returns:
        Formatted string representation
    """
    if hasattr(result, "__dict__"):
        formatted_parts = []
        for key, value in result.__dict__.items():
            if not key.startswith("_"):
                formatted_parts.append(f"**{key}:** {value}")
        return "\n".join(formatted_parts)
    else:
        return str(result)


def create_metrics_chart(
    metrics_data: List[Dict[str, Any]],
    x_field: str = "step",
    y_fields: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create a metrics visualization chart.

    Args:
        metrics_data: List of metrics dictionaries
        x_field: Field to use for x-axis
        y_fields: Fields to plot on y-axis

    Returns:
        Plotly figure
    """
    df = pd.DataFrame(metrics_data)

    if y_fields is None:
        y_fields = [
            col
            for col in df.columns
            if col != x_field and pd.api.types.is_numeric_dtype(df[col])
        ]

    fig = go.Figure()

    for y_field in y_fields:
        if y_field in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_field],
                    y=df[y_field],
                    mode="lines+markers",
                    name=y_field,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title="Metrics Over Time",
        xaxis_title=x_field.replace("_", " ").title(),
        yaxis_title="Value",
        hovermode="x unified",
        showlegend=True,
    )

    return fig
