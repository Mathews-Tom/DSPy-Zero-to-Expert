# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore marimo dspy

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import logging
    import sys
    import time
    from contextlib import contextmanager
    from dataclasses import dataclass, field
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional

    import dspy
    import marimo as mo
    from marimo import output

    from common.utils import get_config

    # Add current directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))

    # Configure logging for tracing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return (
        Any,
        Optional,
        cleandoc,
        contextmanager,
        dataclass,
        dspy,
        field,
        get_config,
        json,
        logger,
        mo,
        output,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # 🔍 DSPy Tracing & Debugging System

            Welcome to the comprehensive tracing and debugging system for DSPy! This interactive tool helps you:

            - **Trace Execution**: Monitor DSPy module execution in real-time
            - **Analyze Performance**: Identify bottlenecks and optimization opportunities
            - **Debug Issues**: Step through execution traces to find problems
            - **Visualize Flow**: See how data flows through your DSPy pipeline
            - **Optimize Performance**: Get actionable suggestions for improvements

            Let's start tracing your DSPy modules! 🚀
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(Any, Optional, contextmanager, dataclass, field, json, logger, time):
    # Tracing System Core Classes
    @dataclass
    class TraceEvent:
        """Represents a single trace event in DSPy execution"""

        event_id: str
        timestamp: float
        event_type: str  # 'module_start', 'module_end', 'prediction', 'error'
        module_name: str
        inputs: dict[str, Any] = field(default_factory=dict)
        outputs: dict[str, Any] = field(default_factory=dict)
        metadata: dict[str, Any] = field(default_factory=dict)
        duration: Optional[float] = None
        error: Optional[str] = None
        parent_id: Optional[str] = None

    @dataclass
    class ExecutionTrace:
        """Complete execution trace for a DSPy pipeline"""

        trace_id: str
        start_time: float
        end_time: Optional[float] = None
        events: list[TraceEvent] = field(default_factory=list)
        total_duration: Optional[float] = None
        success: bool = True
        error_count: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

        def add_event(self, event: TraceEvent):
            """Add an event to the trace"""
            self.events.append(event)
            if event.event_type == "error":
                self.error_count += 1
                self.success = False

        def finalize(self):
            """Finalize the trace with end time and duration"""
            self.end_time = time.time()
            if self.start_time:
                self.total_duration = self.end_time - self.start_time

        def to_dict(self) -> dict[str, Any]:
            """Convert trace to dictionary for serialization"""
            return {
                "trace_id": self.trace_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": self.total_duration,
                "success": self.success,
                "error_count": self.error_count,
                "metadata": self.metadata,
                "events": [
                    {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "module_name": event.module_name,
                        "inputs": event.inputs,
                        "outputs": event.outputs,
                        "metadata": event.metadata,
                        "duration": event.duration,
                        "error": event.error,
                        "parent_id": event.parent_id,
                    }
                    for event in self.events
                ],
            }

    class DSPyTracer:
        """Main tracing system for DSPy modules"""

        def __init__(self):
            self.current_trace: Optional[ExecutionTrace] = None
            self.traces: list[ExecutionTrace] = []
            self.event_counter = 0

        def start_trace(
            self, trace_id: str, metadata: Optional[dict] = None
        ) -> ExecutionTrace:
            """Start a new execution trace"""
            self.current_trace = ExecutionTrace(
                trace_id=trace_id,
                start_time=time.time(),
                metadata=metadata or {},
            )
            logger.info(f"Started trace: {trace_id}")
            return self.current_trace

        def end_trace(self):
            """End the current trace"""
            if self.current_trace:
                self.current_trace.finalize()
                self.traces.append(self.current_trace)
                logger.info(
                    f"Ended trace: {self.current_trace.trace_id} "
                    f"(Duration: {self.current_trace.total_duration:.3f}s)"
                )
                self.current_trace = None

        def add_event(
            self,
            event_type: str,
            module_name: str,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            metadata: Optional[dict] = None,
            error: Optional[str] = None,
            parent_id: Optional[str] = None,
        ) -> str:
            """Add an event to the current trace"""
            if not self.current_trace:
                logger.warning("No active trace - creating default trace")
                self.start_trace("default_trace")

            self.event_counter += 1
            event_id = f"event_{self.event_counter}"

            event = TraceEvent(
                event_id=event_id,
                timestamp=time.time(),
                event_type=event_type,
                module_name=module_name,
                inputs=inputs or {},
                outputs=outputs or {},
                metadata=metadata or {},
                error=error,
                parent_id=parent_id,
            )

            self.current_trace.add_event(event)
            return event_id

        @contextmanager
        def trace_module(self, module_name: str, inputs: Optional[dict] = None):
            """Context manager for tracing module execution"""
            start_event_id = self.add_event(
                "module_start", module_name, inputs=inputs or {}
            )
            start_time = time.time()

            try:
                yield start_event_id
                # Module executed successfully
                duration = time.time() - start_time
                self.add_event(
                    "module_end",
                    module_name,
                    metadata={"duration": duration},
                    parent_id=start_event_id,
                )
            except Exception as e:
                # Module execution failed
                duration = time.time() - start_time
                self.add_event(
                    "error",
                    module_name,
                    error=str(e),
                    metadata={"duration": duration},
                    parent_id=start_event_id,
                )
                raise

        def get_trace_summary(self, trace_id: str) -> Optional[dict[str, Any]]:
            """Get summary statistics for a trace"""
            trace = next((t for t in self.traces if t.trace_id == trace_id), None)
            if not trace:
                return None

            module_stats = {}
            for event in trace.events:
                if event.module_name not in module_stats:
                    module_stats[event.module_name] = {
                        "calls": 0,
                        "total_duration": 0,
                        "errors": 0,
                    }

                if event.event_type == "module_end":
                    module_stats[event.module_name]["calls"] += 1
                    if event.duration:
                        module_stats[event.module_name][
                            "total_duration"
                        ] += event.duration
                elif event.event_type == "error":
                    module_stats[event.module_name]["errors"] += 1

            return {
                "trace_id": trace.trace_id,
                "total_duration": trace.total_duration,
                "success": trace.success,
                "error_count": trace.error_count,
                "total_events": len(trace.events),
                "module_stats": module_stats,
            }

        def export_trace(self, trace_id: str, filepath: str):
            """Export trace to JSON file"""
            trace = next((t for t in self.traces if t.trace_id == trace_id), None)
            if not trace:
                raise ValueError(f"Trace {trace_id} not found")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2)

    # Initialize global tracer
    tracer = DSPyTracer()

    return (tracer,)


@app.cell
def _(mo, output):
    # Configuration Section
    cell3_desc = mo.md("## ⚙️ Step 1: Configure Tracing")

    trace_name = mo.ui.text(
        value="dspy_execution_trace",
        label="Trace Name",
        placeholder="Enter a name for your trace",
    )

    enable_detailed = mo.ui.checkbox(
        value=True, label="Enable detailed input/output logging"
    )

    enable_performance = mo.ui.checkbox(
        value=True, label="Enable performance metrics collection"
    )

    enable_errors = mo.ui.checkbox(
        value=True, label="Enable error tracking and analysis"
    )

    cell3_content = mo.vstack(
        [
            trace_name,
            mo.md("### Tracing Options"),
            enable_detailed,
            enable_performance,
            enable_errors,
        ]
    )

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (
        enable_detailed,
        enable_errors,
        enable_performance,
        trace_name,
    )


@app.cell
def _(get_config, mo, output):
    # DSPy Module Setup
    cell4_desc = mo.md("## 🧠 Step 2: Configure DSPy Module")

    # Get available LLM providers
    try:
        config = get_config()
        cell4_available_providers = config.get_available_llm_providers()
    except Exception as _:
        cell4_available_providers = ["openai", "anthropic", "local"]

    provider_selector = mo.ui.dropdown(
        options=cell4_available_providers,
        value=cell4_available_providers[0] if cell4_available_providers else "openai",
        label="Select LLM Provider",
    )

    module_type = mo.ui.dropdown(
        options=["Predict", "ChainOfThought", "ReAct"],
        value="Predict",
        label="Select DSPy Module Type",
    )

    cell4_content = mo.vstack([provider_selector, module_type])

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return module_type, provider_selector


@app.cell
def _(dspy, module_type, provider_selector):
    # Initialize DSPy Module
    module = None
    signature = None

    if provider_selector.value and module_type.value:
        try:
            # Configure DSPy with selected provider
            if provider_selector.value == "openai":
                lm = dspy.LM("openai/gpt-5")
            elif provider_selector.value == "anthropic":
                lm = dspy.LM("anthropic/claude-3-7-sonnet-20250219")
            else:
                lm = dspy.LM(
                    "openai/gpt-3.5-turbo", api_base="http://localhost:1234/v1"
                )

            dspy.configure(lm=lm)

            # Create signature
            class TracingSignature(dspy.Signature):
                """Analyze the given input and provide insights"""

                input_text = dspy.InputField(desc="Text to analyze")
                analysis = dspy.OutputField(desc="Detailed analysis of the input")

            signature = TracingSignature

            # Create module based on selection
            if module_type.value == "Predict":
                module = dspy.Predict(TracingSignature)
            elif module_type.value == "ChainOfThought":
                module = dspy.ChainOfThought(TracingSignature)
            else:  # ReAct
                module = dspy.ReAct(TracingSignature)

        except Exception as e:
            print(f"Error configuring DSPy module: {e}")
            module = None
            signature = None

    return (module,)


@app.cell
def _(cleandoc, mo, module, output):
    # Module Status Display
    if module:
        cell6_out = mo.md(
            cleandoc(
                f"""
                ✅ **DSPy Module Configured Successfully**  
                - Module Type: {type(module).__name__}  
                - Signature: TracingSignature  
                - Ready for tracing execution  
                """
            )
        )
    else:
        cell6_out = mo.md("⚠️ Configure DSPy module above to continue")

    output.replace(cell6_out)
    return


@app.cell
def _(mo, module, output):
    # Input Section
    if not module:
        cell7_desc = mo.md("*Configure DSPy module first*")
        cell7_content = mo.md("")
        input_text = None
        execute_button = None
    else:
        cell7_desc = mo.md("## 📝 Step 3: Execute with Tracing")

        input_text = mo.ui.text_area(
            placeholder="Enter text to analyze...",
            label="Input Text",
            rows=4,
            value="Machine learning is transforming how we solve complex problems.",
        )

        execute_button = mo.ui.run_button(label="Execute with Tracing", kind="success")

        cell7_content = mo.vstack([input_text, execute_button])

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return (
        execute_button,
        input_text,
    )


@app.cell
def _(
    enable_detailed,
    enable_errors,
    enable_performance,
    execute_button,
    input_text,
    module,
    time,
    trace_name,
    tracer,
):
    # Execute Module with Tracing
    execution_result = None
    trace_data = None

    if (
        execute_button is not None
        and execute_button.value
        and module
        and input_text is not None
        and input_text.value
    ):
        try:
            # Start tracing
            trace_metadata = {
                "detailed_logging": enable_detailed.value,
                "performance_metrics": enable_performance.value,
                "error_tracking": enable_errors.value,
            }

            trace = tracer.start_trace(trace_name.value, trace_metadata)

            # Execute with tracing
            with tracer.trace_module(
                type(module).__name__, {"input_text": input_text.value}
            ):
                # Add prediction event
                start_time = time.time()
                result = module(input_text=input_text.value)
                duration = time.time() - start_time

                # Log prediction details if enabled
                if enable_detailed.value:
                    tracer.add_event(
                        "prediction",
                        type(module).__name__,
                        inputs={"input_text": input_text.value},
                        outputs={"analysis": result.analysis},
                        metadata={"prediction_duration": duration},
                    )

            # End tracing
            tracer.end_trace()

            execution_result = result
            trace_data = tracer.get_trace_summary(trace_name.value)

        except Exception as e:
            # Handle execution error
            if tracer.current_trace:
                tracer.add_event("error", type(module).__name__, error=str(e))
                tracer.end_trace()

            execution_result = {"error": str(e)}
            trace_data = tracer.get_trace_summary(trace_name.value)

    return (
        execution_result,
        trace_data,
    )


@app.cell
def _(cleandoc, execution_result, mo, output):
    # Display Execution Results
    if not execution_result:
        cell9_out = mo.md("*Execute the module above to see results*")
    else:
        if "error" in execution_result:
            cell9_desc = mo.md("## ❌ Execution Error")
            cell9_content = mo.md(f"**Error:** {execution_result['error']}")
        else:
            cell9_desc = mo.md("## ✅ Execution Results")
            cell9_content = mo.md(
                cleandoc(
                    f"""
                    **Analysis Result:**  
                    {execution_result.analysis}
                    """
                )
            )

        cell9_out = mo.vstack([cell9_desc, cell9_content])

    output.replace(cell9_out)
    return


@app.cell
def _(cleandoc, mo, output, trace_data):
    # Display Trace Analysis
    if not trace_data:
        cell10_out = mo.md("*No trace data available*")
    else:
        # Create trace summary
        _trace_summary = cleandoc(
            f"""
            ## 📊 Trace Analysis

            ### Execution Summary
            - **Trace ID:** {trace_data['trace_id']}
            - **Total Duration:** {trace_data['total_duration']:.3f} seconds
            - **Success:** {'✅ Yes' if trace_data['success'] else '❌ No'}
            - **Total Events:** {trace_data['total_events']}
            - **Error Count:** {trace_data['error_count']}

            ### Module Performance
            """
        )

        # Add module statistics
        for _name, _stats in trace_data["module_stats"].items():
            _avg_duration = (
                _stats["total_duration"] / _stats["calls"] if _stats["calls"] > 0 else 0
            )
            _trace_summary += cleandoc(
                f"""
                **{_name}:**
                - Calls: {_stats['calls']}
                - Total Duration: {_stats['total_duration']:.3f}s
                - Average Duration: {_avg_duration:.3f}s
                - Errors: {_stats['errors']}
                """
            )

        cell10_out = mo.md(_trace_summary)

    output.replace(cell10_out)
    return


@app.cell
def _(mo, output, trace_data):
    # Performance Optimization Suggestions
    if not trace_data:
        cell11_out = mo.md("*No trace data available for optimization analysis*")
    else:
        cell11_suggestions = []

        # Analyze performance and generate suggestions
        total_duration = trace_data.get("total_duration", 0)
        error_count = trace_data.get("error_count", 0)

        if total_duration > 5.0:
            cell11_suggestions.append(
                "⚡ **Slow Execution**: Consider using a faster model or optimizing prompts"
            )

        if error_count > 0:
            cell11_suggestions.append(
                "🐛 **Errors Detected**: Review error logs and improve error handling"
            )

        # Check module-specific performance
        for _name, _stats in trace_data["module_stats"].items():
            if _stats["calls"] > 0:
                _avg_duration = _stats["total_duration"] / _stats["calls"]
                if _avg_duration > 3.0:
                    cell11_suggestions.append(
                        f"🔧 **{_name}**: Average duration ({_avg_duration:.2f}s) is high - consider prompt optimization"
                    )

        if not cell11_suggestions:
            cell11_suggestions.append(
                "✅ **Good Performance**: No optimization suggestions at this time"
            )

        cell11_desc = mo.md("## 💡 Performance Optimization Suggestions")
        cell11_content = mo.md(
            "\n".join(f"- {suggestion}" for suggestion in cell11_suggestions)
        )

        cell11_out = mo.vstack([cell11_desc, cell11_content])

    output.replace(cell11_out)
    return


@app.cell
def _(mo, output, tracer, trace_name):
    # Export Trace Data
    if not tracer.traces:
        cell12_desc = mo.md("*No traces available for export*")
        cell12_content = mo.md("")
        cell12_export_button = None
    else:
        cell12_desc = mo.md("## 💾 Step 4: Export Trace Data")

        cell12_filename = mo.ui.text(
            value=f"{trace_name.value}_trace.json",
            label="Export Filename",
        )

        cell12_export_button = mo.ui.button(label="Export Trace", kind="primary")

        cell12_content = mo.vstack([cell12_filename, cell12_export_button])

    cell12_out = mo.vstack([cell12_desc, cell12_content])
    output.replace(cell12_out)
    return (cell12_export_button,)


@app.cell
def _(cell12_export_button, mo, output, trace_name, tracer):
    # Handle Trace Export
    if cell12_export_button and cell12_export_button.value:
        try:
            _export_filename = f"{trace_name.value}_trace.json"
            tracer.export_trace(trace_name.value, _export_filename)
            cell13_out = mo.md(
                f"✅ **Trace exported successfully to {_export_filename}**"
            )
        except Exception as e:
            cell13_out = mo.md(f"❌ **Export failed:** {str(e)}")
    else:
        cell13_out = mo.md("*Click 'Export Trace' to save trace data*")

    output.replace(cell13_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with tips and next steps
    cell14_out = mo.md(
        cleandoc(
            """
            ## 🎯 Tracing Best Practices

            ### ✅ What You've Learned

            - **Execution Tracing**: Monitor DSPy module execution in real-time
            - **Performance Analysis**: Identify bottlenecks and optimization opportunities
            - **Error Tracking**: Capture and analyze execution errors
            - **Data Export**: Save trace data for further analysis

            ### 💡 Pro Tips

            1. **Use Descriptive Trace Names**: Make it easy to identify traces later
            2. **Enable Detailed Logging**: For debugging, capture full input/output data
            3. **Monitor Performance**: Watch for slow modules and optimize accordingly
            4. **Export Important Traces**: Save traces for comparison and analysis
            5. **Analyze Patterns**: Look for recurring issues or performance bottlenecks

            ### 🚀 Next Steps

            - **Debugging Dashboard**: Use the debugging utilities for deeper analysis
            - **Observability Dashboard**: Monitor system health and performance
            - **Performance Monitor**: Set up continuous performance monitoring

            **Ready to debug?** Try the debugging utilities notebook next! 🔧
            """
        )
    )

    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
