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
    import traceback
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

    # Configure logging for debugging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    return (
        Any,
        Optional,
        cleandoc,
        dataclass,
        dspy,
        field,
        get_config,
        json,
        logger,
        mo,
        output,
        traceback,
    )


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # 🔧 DSPy Debugging Utilities

            Welcome to the comprehensive debugging toolkit for DSPy! This interactive system provides:

            - **Step-by-Step Debugging**: Execute modules with detailed inspection
            - **Interactive Breakpoints**: Pause execution to examine state
            - **Variable Inspector**: Deep dive into module inputs and outputs
            - **Error Analysis**: Comprehensive error diagnosis and suggestions
            - **Execution Flow Visualization**: See how data flows through your pipeline
            - **Performance Profiling**: Identify bottlenecks and optimization opportunities

            Let's debug your DSPy modules like a pro! 🕵️‍♂️
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(Any, Optional, dataclass, field, logger, traceback):
    # Debugging System Core Classes
    @dataclass
    class DebugPoint:
        """Represents a debugging breakpoint"""

        point_id: str
        module_name: str
        condition: Optional[str] = None
        enabled: bool = True
        hit_count: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ExecutionState:
        """Captures the execution state at a debug point"""

        point_id: str
        timestamp: float
        module_name: str
        inputs: dict[str, Any] = field(default_factory=dict)
        outputs: dict[str, Any] = field(default_factory=dict)
        local_vars: dict[str, Any] = field(default_factory=dict)
        stack_trace: list[str] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class DebugSession:
        """Complete debugging session"""

        session_id: str
        start_time: float
        breakpoints: list[DebugPoint] = field(default_factory=list)
        execution_states: list[ExecutionState] = field(default_factory=list)
        errors: list[dict[str, Any]] = field(default_factory=list)
        performance_data: dict[str, Any] = field(default_factory=dict)

    class DSPyDebugger:
        """Main debugging system for DSPy modules"""

        def __init__(self):
            self.current_session: Optional[DebugSession] = None
            self.sessions: list[DebugSession] = []
            self.breakpoints: dict[str, DebugPoint] = {}
            self.step_mode: bool = False
            self.continue_execution: bool = True

        def start_session(self, session_id: str) -> DebugSession:
            """Start a new debugging session"""
            import time

            self.current_session = DebugSession(
                session_id=session_id, start_time=time.time()
            )
            logger.info(f"Started debug session: {session_id}")
            return self.current_session

        def end_session(self):
            """End the current debugging session"""
            if self.current_session:
                self.sessions.append(self.current_session)
                logger.info(f"Ended debug session: {self.current_session.session_id}")
                self.current_session = None

        def add_breakpoint(
            self,
            point_id: str,
            module_name: str,
            condition: Optional[str] = None,
        ) -> DebugPoint:
            """Add a breakpoint"""
            break_point = DebugPoint(
                point_id=point_id, module_name=module_name, condition=condition
            )
            self.breakpoints[point_id] = break_point
            if self.current_session:
                self.current_session.breakpoints.append(break_point)
            logger.info(f"Added breakpoint: {point_id} for {module_name}")
            return breakpoint

        def remove_breakpoint(self, point_id: str):
            """Remove a breakpoint"""
            if point_id in self.breakpoints:
                del self.breakpoints[point_id]
                logger.info(f"Removed breakpoint: {point_id}")

        def should_break(self, point_id: str, context: dict[str, Any]) -> bool:
            """Check if execution should break at this point"""
            if point_id not in self.breakpoints:
                return False

            break_point = self.breakpoints[point_id]
            if not breakpoint.enabled:
                return False

            # Check condition if specified
            if breakpoint.condition:
                try:
                    # Simple condition evaluation (can be enhanced)
                    return eval(break_point.condition, {"__builtins__": {}}, context)
                except Exception as e:
                    logger.warning(f"Breakpoint condition error: {e}")
                    return False

            return True

        def capture_state(
            self,
            point_id: str,
            module_name: str,
            inputs: Optional[dict] = None,
            outputs: Optional[dict] = None,
            local_vars: Optional[dict] = None,
        ) -> ExecutionState:
            """Capture execution state at a debug point"""
            import time

            state = ExecutionState(
                point_id=point_id,
                timestamp=time.time(),
                module_name=module_name,
                inputs=inputs or {},
                outputs=outputs or {},
                local_vars=local_vars or {},
                stack_trace=traceback.format_stack(),
            )

            if self.current_session:
                self.current_session.execution_states.append(state)

            # Update breakpoint hit count
            if point_id in self.breakpoints:
                self.breakpoints[point_id].hit_count += 1

            return state

        def analyze_error(
            self, error: Exception, context: dict[str, Any]
        ) -> dict[str, Any]:
            """Analyze an error and provide debugging suggestions"""
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context": context,
                "suggestions": [],
            }

            # Generate suggestions based on error type
            if isinstance(error, AttributeError):
                error_info["suggestions"].append(
                    "Check if the object has the expected attributes"
                )
                error_info["suggestions"].append(
                    "Verify that the module was properly initialized"
                )
            elif isinstance(error, KeyError):
                error_info["suggestions"].append(
                    "Check if all required keys are present in the input"
                )
                error_info["suggestions"].append(
                    "Verify the signature field names match the input data"
                )
            elif isinstance(error, ValueError):
                error_info["suggestions"].append("Check input data types and formats")
                error_info["suggestions"].append(
                    "Verify that input values are within expected ranges"
                )
            elif "API" in str(error) or "rate limit" in str(error).lower():
                error_info["suggestions"].append(
                    "Check API key configuration and rate limits"
                )
                error_info["suggestions"].append(
                    "Consider adding retry logic or reducing request frequency"
                )

            if self.current_session:
                self.current_session.errors.append(error_info)

            return error_info

        def get_variable_info(self, var_name: str, value: Any) -> dict[str, Any]:
            """Get detailed information about a variable"""
            return {
                "name": var_name,
                "type": type(value).__name__,
                "value": str(value)[:500],  # Truncate long values
                "size": len(str(value)),
                "is_callable": callable(value),
                "attributes": [attr for attr in dir(value) if not attr.startswith("_")],
            }

        def profile_execution(self, func, *args, **kwargs):
            """Profile function execution"""
            import time

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                profile_data = {
                    "function": (
                        func.__name__ if hasattr(func, "__name__") else str(func)
                    ),
                    "duration": duration,
                    "success": True,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                }

                if self.current_session:
                    self.current_session.performance_data[
                        func.__name__ if hasattr(func, "__name__") else "unknown"
                    ] = profile_data

                return result, profile_data
            except Exception as e:
                duration = time.time() - start_time
                profile_data = {
                    "function": (
                        func.__name__ if hasattr(func, "__name__") else str(func)
                    ),
                    "duration": duration,
                    "success": False,
                    "error": str(e),
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                }

                if self.current_session:
                    self.current_session.performance_data[
                        func.__name__ if hasattr(func, "__name__") else "unknown"
                    ] = profile_data

                raise

    # Initialize global debugger
    debugger = DSPyDebugger()

    return (debugger,)


@app.cell
def _(mo, output):
    # Debug Session Configuration
    cell3_desc = mo.md("## 🎯 Step 1: Configure Debug Session")

    session_name = mo.ui.text(
        value="dspy_debug_session",
        label="Debug Session Name",
        placeholder="Enter a name for your debug session",
    )

    step_mode = mo.ui.checkbox(value=False, label="Enable step-by-step execution")

    capture_vars = mo.ui.checkbox(
        value=True, label="Capture local variables at breakpoints"
    )

    profile_performance = mo.ui.checkbox(
        value=True, label="Enable performance profiling"
    )

    cell3_content = mo.vstack(
        [
            session_name,
            mo.md("### Debug Options"),
            step_mode,
            capture_vars,
            profile_performance,
        ]
    )

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return capture_vars, profile_performance, session_name


@app.cell
def _(get_config, mo, output):
    # DSPy Module Setup for Debugging
    cell4_desc = mo.md("## 🧠 Step 2: Configure DSPy Module for Debugging")

    # Get available LLM providers
    try:
        config = get_config()
        available_providers = config.get_available_llm_providers()
    except Exception:
        available_providers = ["openai", "anthropic", "local"]

    provider_selector = mo.ui.dropdown(
        options=available_providers,
        value=available_providers[0] if available_providers else "openai",
        label="Select LLM Provider",
    )

    module_type = mo.ui.dropdown(
        options=["Predict", "ChainOfThought", "ReAct"],
        value="ChainOfThought",
        label="Select DSPy Module Type",
    )

    cell4_content = mo.vstack([provider_selector, module_type])

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return module_type, provider_selector


@app.cell
def _(dspy, module_type, provider_selector):
    # Initialize DSPy Module for Debugging
    debug_module = None
    cell5_debug_signature = None

    if provider_selector.value and module_type.value:
        try:
            # Configure DSPy with selected provider
            if provider_selector.value == "openai":
                lm = dspy.LM("openai/gpt-3.5-turbo")
            elif provider_selector.value == "anthropic":
                lm = dspy.LM("anthropic/claude-3-haiku-20240307")
            else:
                lm = dspy.LM(
                    "openai/gpt-3.5-turbo", api_base="http://localhost:1234/v1"
                )

            dspy.configure(lm=lm)

            # Create signature for debugging
            class DebuggingSignature(dspy.Signature):
                """Process the input and provide detailed reasoning"""

                question = dspy.InputField(desc="Question or problem to solve")
                context = dspy.InputField(desc="Additional context (optional)")
                answer = dspy.OutputField(desc="Detailed answer with reasoning")

            cell5_debug_signature = DebuggingSignature

            # Create module based on selection
            if module_type.value == "Predict":
                debug_module = dspy.Predict(DebuggingSignature)
            elif module_type.value == "ChainOfThought":
                debug_module = dspy.ChainOfThought(DebuggingSignature)
            else:  # ReAct
                debug_module = dspy.ReAct(DebuggingSignature)

        except Exception as _:
            pass

    return (debug_module,)


@app.cell
def _(cleandoc, debug_module, mo, output):
    # Module Status Display
    if debug_module:
        cell6_out = mo.md(
            cleandoc(
                f"""
                ✅ **DSPy Module Ready for Debugging**
                - Module Type: {type(debug_module).__name__}
                - Signature: DebuggingSignature
                - Ready for step-by-step debugging
                """
            )
        )
    else:
        cell6_out = mo.md("⚠️ Configure DSPy module above to continue")

    output.replace(cell6_out)
    return


@app.cell
def _(debug_module, mo, output):
    # Breakpoint Configuration
    if not debug_module:
        cell7_desc = mo.md("*Configure DSPy module first*")
        cell7_content = mo.md("")
        add_breakpoint_button = None
    else:
        cell7_desc = mo.md("## 🔴 Step 3: Configure Breakpoints")

        breakpoint_name = mo.ui.text(
            value="module_start",
            label="Breakpoint Name",
            placeholder="Enter breakpoint identifier",
        )

        breakpoint_condition = mo.ui.text(
            value="",
            label="Condition (optional)",
            placeholder="e.g., len(question) > 10",
        )

        add_breakpoint_button = mo.ui.run_button(label="Add Breakpoint", kind="info")

        cell7_content = mo.vstack(
            [breakpoint_name, breakpoint_condition, add_breakpoint_button]
        )

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return add_breakpoint_button, breakpoint_condition, breakpoint_name


@app.cell
def _(
    add_breakpoint_button,
    breakpoint_condition,
    breakpoint_name,
    debugger,
    module_type,
):
    # Handle Breakpoint Addition
    breakpoints = []

    if (
        add_breakpoint_button is not None
        and add_breakpoint_button.value
        and breakpoint_name.value
    ):
        try:
            condition = (
                breakpoint_condition.value if breakpoint_condition.value else None
            )
            _breakpoint = debugger.add_breakpoint(
                breakpoint_name.value,
                module_type.value,
                condition=condition,
            )
            breakpoints = list(debugger.breakpoints.values())
        except Exception as _:
            pass

    return (breakpoints,)


@app.cell
def _(breakpoints, mo, output):
    # Display Active Breakpoints
    if not breakpoints:
        cell9_out = mo.md("*No breakpoints configured*")
    else:
        cell9_desc = mo.md("### Active Breakpoints")

        _breakpoint_list = []
        for bp in breakpoints:
            _condition_text = f" (Condition: {bp.condition})" if bp.condition else ""
            _status = "🟢 Enabled" if bp.enabled else "🔴 Disabled"
            _breakpoint_list.append(
                f"- **{bp.point_id}** - {bp.module_name}{_condition_text} - {_status} (Hits: {bp.hit_count})"
            )

        cell9_content = mo.md("\n".join(_breakpoint_list))
        cell9_out = mo.vstack([cell9_desc, cell9_content])

    output.replace(cell9_out)
    return


@app.cell
def _(debug_module, mo, output):
    # Debug Input Section
    if not debug_module:
        cell10_desc = mo.md("*Configure DSPy module first*")
        cell10_content = mo.md("")
        question_input = None
        context_input = None
        debug_execute = None
    else:
        cell10_desc = mo.md("## 🐛 Step 4: Execute with Debugging")

        question_input = mo.ui.text_area(
            placeholder="Enter your question...",
            label="Question",
            rows=3,
            value="What are the key principles of machine learning?",
        )

        context_input = mo.ui.text_area(
            placeholder="Enter additional context (optional)...",
            label="Context",
            rows=2,
            value="Focus on supervised learning approaches.",
        )

        debug_execute = mo.ui.run_button(label="Execute with Debugging", kind="success")

        cell10_content = mo.vstack([question_input, context_input, debug_execute])

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return context_input, debug_execute, question_input


@app.cell
def _(
    capture_vars,
    context_input,
    debug_execute,
    debug_module,
    debugger,
    profile_performance,
    question_input,
    session_name,
):
    # Execute Module with Debugging
    debug_result = None
    debug_states = []
    debug_errors = []

    if (
        debug_execute is not None
        and debug_execute.value
        and debug_module
        and question_input
        and question_input.value
    ):
        try:
            # Start debug session
            session = debugger.start_session(session_name.value)

            # Prepare inputs
            _inputs = {
                "question": question_input.value,
                "context": (context_input.value if context_input.value else ""),
            }

            # Check for breakpoints before execution
            if debugger.should_break("module_start", _inputs):
                _state = debugger.capture_state(
                    "module_start",
                    type(debug_module).__name__,
                    inputs=_inputs,
                    local_vars=(
                        {"module_type": type(debug_module).__name__}
                        if capture_vars.value
                        else None
                    ),
                )
                debug_states.append(_state)

            # Execute with profiling if enabled
            if profile_performance.value:
                result, profile_data = debugger.profile_execution(
                    debug_module, **_inputs
                )
            else:
                result = debug_module(**_inputs)

            # Check for breakpoints after execution
            if debugger.should_break("module_end", {"result": result.answer}):
                state = debugger.capture_state(
                    "module_end",
                    type(debug_module).__name__,
                    outputs={"answer": result.answer},
                    local_vars=(
                        {"result_length": len(result.answer)}
                        if capture_vars.value
                        else None
                    ),
                )
                debug_states.append(state)

            debug_result = result
            debugger.end_session()

        except Exception as e:
            # Analyze error
            error_analysis = debugger.analyze_error(e, _inputs)
            debug_errors.append(error_analysis)
            debugger.end_session()

    return debug_errors, debug_result, debug_states


@app.cell
def _(cleandoc, debug_errors, debug_result, mo, output):
    # Display Debug Results
    if debug_errors:
        cell12_desc = mo.md("## ❌ Debug Results - Error Detected")

        _error = debug_errors[0]
        _suggestions_text = "\n".join(
            f"- {suggestion}" for suggestion in _error["suggestions"]
        )

        cell12_content = mo.md(
            cleandoc(
                f"""
                **Error Type:** {_error['type']}

                **Error Message:** {_error['message']}

                **Debugging Suggestions:**
                {_suggestions_text}
                """
            )
        )

        cell12_out = mo.vstack([cell12_desc, cell12_content])
    elif debug_result:
        cell12_desc = mo.md("## ✅ Debug Results - Successful Execution")
        cell12_content = mo.md(
            cleandoc(
                f"""
                **Answer:**
                {debug_result.answer}
                """
            )
        )
        cell12_out = mo.vstack([cell12_desc, cell12_content])
    else:
        cell12_out = mo.md("*Execute the module above to see debug results*")

    output.replace(cell12_out)
    return


@app.cell
def _(cleandoc, debug_states, json, mo, output):
    # Display Debug States
    if not debug_states:
        cell13_out = mo.md("*No debug states captured*")
    else:
        cell13_desc = mo.md("## 🔍 Debug State Analysis")

        _states_content = []
        for i, _state in enumerate(debug_states, 1):
            _state_info = cleandoc(
                f"""
                ### State {i}: {_state.point_id}

                **Module:** {_state.module_name}
                **Timestamp:** {_state.timestamp:.3f}

                **Inputs:**
                {json.dumps(_state.inputs, indent=2) if _state.inputs else "None"}

                **Outputs:**
                {json.dumps(_state.outputs, indent=2) if _state.outputs else "None"}

                **Local Variables:**
                {json.dumps(_state.local_vars, indent=2) if _state.local_vars else "None"}
                """
            )
            _states_content.append(_state_info)

        cell13_content = mo.md("\n\n---\n\n".join(_states_content))
        cell13_out = mo.vstack([cell13_desc, cell13_content])

    output.replace(cell13_out)
    return


@app.cell
def _(cleandoc, debugger, mo, output):
    # Performance Analysis
    if not debugger.sessions or not debugger.sessions[-1].performance_data:
        cell14_out = mo.md("*No performance data available*")
    else:
        cell14_desc = mo.md("## ⚡ Performance Analysis")

        _perf_data = debugger.sessions[-1].performance_data
        _perf_content = []

        for func_name, data in _perf_data.items():
            _status = "✅ Success" if data["success"] else "❌ Failed"
            _perf_info = cleandoc(
                f"""
                **Function:** {func_name}
                - Duration: {data['duration']:.3f} seconds
                - Status: {_status}
                - Arguments: {data['args_count']} args, {data['kwargs_count']} kwargs
                """
            )
            if not data["success"]:
                _perf_info += f"- Error: {data.get('error', 'Unknown error')}\n"

            _perf_content.append(_perf_info)

        cell14_content = mo.md("\n".join(_perf_content))
        cell14_out = mo.vstack([cell14_desc, cell14_content])

    output.replace(cell14_out)
    return


@app.cell
def _(debugger, mo, output):
    # Variable Inspector
    if not debugger.sessions:
        cell15_desc = mo.md("*No debug session data available*")
        cell15_content = mo.md("")
        var_name = None
        inspect_button = None
    else:
        cell15_desc = mo.md("## 🔬 Variable Inspector")

        var_name = mo.ui.text(
            value="question",
            label="Variable Name to Inspect",
            placeholder="Enter variable name",
        )

        inspect_button = mo.ui.button(label="Inspect Variable", kind="primary")

        cell15_content = mo.vstack([var_name, inspect_button])

    cell15_out = mo.vstack([cell15_desc, cell15_content])
    output.replace(cell15_out)
    return inspect_button, var_name


@app.cell
def _(cleandoc, debugger, inspect_button, mo, output, var_name):
    # Handle Variable Inspection
    if (
        inspect_button is not None
        and inspect_button.value
        and var_name
        and var_name.value
        and debugger.sessions
    ):
        _var_name = var_name.value
        _found_vars = []

        # Search through debug states for the variable
        for _state in debugger.sessions[-1].execution_states:
            if _var_name in _state.inputs:
                _var_info = debugger.get_variable_info(
                    _var_name, _state.inputs[_var_name]
                )
                _var_info["location"] = f"inputs at {_state.point_id}"
                _found_vars.append(_var_info)
            if _var_name in _state.outputs:
                _var_info = debugger.get_variable_info(
                    _var_name, _state.outputs[_var_name]
                )
                _var_info["location"] = f"outputs at {_state.point_id}"
                _found_vars.append(_var_info)
            if _var_name in _state.local_vars:
                _var_info = debugger.get_variable_info(
                    _var_name, _state.local_vars[_var_name]
                )
                _var_info["location"] = f"local vars at {_state.point_id}"
                _found_vars.append(_var_info)

        if _found_vars:
            cell16_desc = mo.md(f"### Variable Inspector Results for '{_var_name}'")

            _var_details = []
            for var_info in _found_vars:
                _detail = cleandoc(
                    f"""
                    **Location:** {var_info['location']}
                    **Type:** {var_info['type']}
                    **Size:** {var_info['size']} characters
                    **Value:** {var_info['value']}
                    **Callable:** {'Yes' if var_info['is_callable'] else 'No'}
                    **Attributes:** {', '.join(var_info['attributes'][:10])}{'...' if len(var_info['attributes']) > 10 else ''}
                    """
                )
                _var_details.append(_detail)

            cell16_content = mo.md("\n---\n".join(_var_details))
            cell16_out = mo.vstack([cell16_desc, cell16_content])
        else:
            cell16_out = mo.md(f"❌ Variable '{_var_name}' not found in debug session")
    else:
        cell16_out = mo.md("*Enter variable name and click 'Inspect Variable'*")

    output.replace(cell16_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with debugging tips
    cell17_out = mo.md(
        cleandoc(
            """
            ---

            ## 🎯 Debugging Best Practices

            ### ✅ What You've Learned

            - **Interactive Debugging**: Step through DSPy module execution
            - **Breakpoint Management**: Set conditional breakpoints for targeted debugging
            - **State Inspection**: Examine inputs, outputs, and local variables
            - **Error Analysis**: Get actionable suggestions for fixing issues
            - **Performance Profiling**: Identify bottlenecks and optimization opportunities

            ### 💡 Pro Tips

            1. **Use Conditional Breakpoints**: Set conditions to break only when specific criteria are met
            2. **Capture Variables**: Enable variable capture to inspect state at breakpoints
            3. **Profile Performance**: Monitor execution time to identify slow operations
            4. **Analyze Errors**: Use error analysis to get specific debugging suggestions
            5. **Step Through Execution**: Use step mode for detailed execution analysis

            ### 🚀 Next Steps

            - **Observability Dashboard**: Monitor system health and performance metrics
            - **Performance Monitor**: Set up continuous performance monitoring
            - **Tracing System**: Use execution tracing for production debugging

            **Ready for observability?** Try the observability dashboard next! 📊
            """
        )
    )

    output.replace(cell17_out)
    return


if __name__ == "__main__":
    app.run()
