# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
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
        json,
        mo,
        output,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🐛 Advanced DSPy Debugging Dashboard

            **Duration:** 60-90 minutes  
            **Prerequisites:** Completed ReAct, Tool Integration, and Multi-Step Reasoning  

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Build comprehensive debugging tools for DSPy agents  
            - ✅ Create interactive debugging interfaces  
            - ✅ Implement step-by-step execution visualization  
            - ✅ Develop performance profiling and optimization tools  
            - ✅ Master advanced error diagnosis and resolution techniques  

            ## 🔍 What is Advanced Debugging?

            Advanced debugging for DSPy agents involves:  
            - **Execution Tracing** - Track every step of agent reasoning  
            - **State Inspection** - Examine agent state at any point  
            - **Performance Profiling** - Identify bottlenecks and optimization opportunities  
            - **Error Analysis** - Diagnose and resolve complex failures  
            - **Interactive Debugging** - Real-time debugging and testing  

            ## 🛠️ Dashboard Features

            Our debugging dashboard includes:  
            1. **Execution Tracer** - Real-time execution monitoring  
            2. **State Inspector** - Deep dive into agent state  
            3. **Performance Profiler** - Speed and resource analysis  
            4. **Error Analyzer** - Intelligent error diagnosis  
            5. **Interactive Debugger** - Live debugging interface  

            Let's build professional debugging tools!
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
                ## ✅ Debugging Dashboard Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  
                - Advanced debugging enabled!  

                Ready to build comprehensive debugging tools!
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
def _(Any, available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🔍 Step 1: Execution Tracer

                Let's build a comprehensive execution tracer for DSPy agents:
                """
            )
        )

        # Execution Tracer Class
        class DSPyExecutionTracer:
            """Advanced execution tracer for DSPy agents."""

            def __init__(self):
                self.traces = []
                self.current_trace = None
                self.trace_id_counter = 0

            def start_trace(self, agent_name: str, input_data: dict):
                """Start a new execution _trace."""
                self.trace_id_counter += 1
                self.current_trace = {
                    "trace_id": self.trace_id_counter,
                    "agent_name": agent_name,
                    "start_time": time.time(),
                    "input_data": input_data,
                    "steps": [],
                    "errors": [],
                    "performance_metrics": {},
                    "final_output": None,
                    "status": "running",
                }
                return self.trace_id_counter

            def log_step(
                self, step_name: str, step_data: dict, step_output: Any = None
            ):
                """Log a step in the current _trace."""
                if not self.current_trace:
                    return

                step_info = {
                    "step_number": len(self.current_trace["steps"]) + 1,
                    "step_name": step_name,
                    "timestamp": time.time(),
                    "step_data": step_data,
                    "step_output": str(step_output) if step_output else None,
                    "execution_time": None,
                }

                self.current_trace["steps"].append(step_info)

            def log_error(
                self, error_type: str, error_message: str, context: dict = None
            ):
                """Log an error in the current _trace."""
                if not self.current_trace:
                    return

                error_info = {
                    "timestamp": time.time(),
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context or {},
                    "step_number": len(self.current_trace["steps"]),
                }

                self.current_trace["errors"].append(error_info)

            def end_trace(self, final_output: Any = None, status: str = "completed"):
                """End the current _trace."""
                if not self.current_trace:
                    return

                self.current_trace["end_time"] = time.time()
                self.current_trace["total_time"] = (
                    self.current_trace["end_time"] - self.current_trace["start_time"]
                )
                self.current_trace["final_output"] = (
                    str(final_output) if final_output else None
                )
                self.current_trace["status"] = status

                # Calculate performance metrics
                self.current_trace["performance_metrics"] = {
                    "total_steps": len(self.current_trace["steps"]),
                    "total_errors": len(self.current_trace["errors"]),
                    "success_rate": (
                        (
                            len(self.current_trace["steps"])
                            - len(self.current_trace["errors"])
                        )
                        / max(len(self.current_trace["steps"]), 1)
                    ),
                    "avg_step_time": (
                        self.current_trace["total_time"]
                        / max(len(self.current_trace["steps"]), 1)
                    ),
                }

                self.traces.append(self.current_trace.copy())
                self.current_trace = None

            def get_trace(self, trace_id: int):
                """Get a specific _trace by ID."""
                for _trace in self.traces:
                    if _trace["trace_id"] == trace_id:
                        return _trace
                return None

            def get_all_traces(self):
                """Get all execution traces."""
                return self.traces

            def analyze_performance(self):
                """Analyze performance across all traces."""
                if not self.traces:
                    return {"error": "No traces available"}

                total_traces = len(self.traces)
                successful_traces = len(
                    [t for t in self.traces if t["status"] == "completed"]
                )

                avg_execution_time = (
                    sum(t.get("total_time", 0) for t in self.traces) / total_traces
                )
                avg_steps = (
                    sum(t["performance_metrics"]["total_steps"] for t in self.traces)
                    / total_traces
                )

                return {
                    "total_traces": total_traces,
                    "successful_traces": successful_traces,
                    "success_rate": successful_traces / total_traces,
                    "avg_execution_time": avg_execution_time,
                    "avg_steps_per_trace": avg_steps,
                    "total_errors": sum(len(t["errors"]) for t in self.traces),
                }

        # Create global tracer
        execution_tracer = DSPyExecutionTracer()

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🔍 Execution Tracer Created

                **Tracer Features:**  
                - **Step-by-Step Logging** - Track every execution step  
                - **Error Tracking** - Capture and analyze errors  
                - **Performance Metrics** - Measure execution performance  
                - **Trace Management** - Store and retrieve execution traces  
                - **Performance Analysis** - Aggregate performance insights  

                The tracer is ready to monitor agent executions!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        DSPyExecutionTracer = None
        execution_tracer = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (execution_tracer,)


@app.cell
def _(available_providers, cleandoc, dspy, execution_tracer, mo, output, time):
    if available_providers and execution_tracer:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🤖 Step 2: Traced Agent Implementation

                Let's create an agent that uses our execution tracer:
                """
            )
        )

        # Traced Agent Signature
        class TracedAgentSignature(dspy.Signature):
            """Agent with comprehensive execution tracing for debugging."""

            task = dspy.InputField(desc="Task or problem to solve")
            context = dspy.InputField(desc="Additional context or constraints")
            analysis = dspy.OutputField(desc="Initial analysis of the task")
            approach = dspy.OutputField(desc="Planned approach to solve the task")
            execution_steps = dspy.OutputField(desc="Step-by-step execution plan")
            result = dspy.OutputField(desc="Final result or solution")
            confidence = dspy.OutputField(desc="Confidence in the result (0.0-1.0)")

        # Traced Agent Class
        class TracedAgent:
            """Agent wrapper with execution tracing."""

            def __init__(self, signature, tracer):
                self.signature = signature
                self.predictor = dspy.ChainOfThought(signature)
                self.tracer = tracer

            def execute(self, **inputs):
                """Execute the agent with full tracing."""
                # Start _trace
                trace_id = self.tracer.start_trace("TracedAgent", inputs)

                try:
                    # Log initial step
                    self.tracer.log_step("initialization", {"inputs": inputs})

                    # Execute prediction
                    self.tracer.log_step(
                        "prediction_start", {"signature": str(self.signature)}
                    )

                    start_time = time.time()
                    result = self.predictor(**inputs)
                    execution_time = time.time() - start_time

                    self.tracer.log_step(
                        "prediction_complete",
                        {
                            "execution_time": execution_time,
                            "result_type": type(result).__name__,
                        },
                        result,
                    )

                    # Log analysis steps
                    if hasattr(result, "analysis"):
                        self.tracer.log_step("analysis", {"analysis": result.analysis})

                    if hasattr(result, "approach"):
                        self.tracer.log_step(
                            "approach_planning", {"approach": result.approach}
                        )

                    if hasattr(result, "execution_steps"):
                        self.tracer.log_step(
                            "execution_planning", {"steps": result.execution_steps}
                        )

                    # End _trace successfully
                    self.tracer.end_trace(result, "completed")

                    return {
                        "success": True,
                        "result": result,
                        "trace_id": trace_id,
                        "execution_time": execution_time,
                    }

                except Exception as e:
                    # Log error and end _trace
                    self.tracer.log_error("execution_error", str(e), {"inputs": inputs})
                    self.tracer.end_trace(None, "failed")

                    return {"success": False, "error": str(e), "trace_id": trace_id}

        # Create traced agent
        traced_agent = TracedAgent(TracedAgentSignature, execution_tracer)

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🤖 Traced Agent Created

                **Agent Features:**  
                - **Full Execution Tracing** - Every step is logged  
                - **Error Handling** - Errors are captured and traced  
                - **Performance Monitoring** - Execution times are tracked  
                - **State Preservation** - All intermediate states are saved  

                The agent is ready for comprehensive debugging!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        TracedAgentSignature = None
        TracedAgent = None
        traced_agent = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (traced_agent,)


@app.cell
def _(available_providers, mo, output, traced_agent):
    if available_providers and traced_agent:
        # Test tasks for the traced agent
        debug_tasks = [
            {
                "task": "Analyze the pros and cons of remote work for software development teams",
                "context": "Consider productivity, collaboration, work-life balance, and company culture",
            },
            {
                "task": "Design a simple recommendation algorithm for a music streaming service",
                "context": "Focus on user preferences, listening history, and collaborative filtering",
            },
            {
                "task": "Plan a marketing strategy for a new eco-friendly product",
                "context": "Target environmentally conscious consumers, limited budget, online focus",
            },
            {
                "task": "Troubleshoot why a web application is running slowly",
                "context": "Recent deployment, increased user load, database performance concerns",
            },
        ]

        task_selector = mo.ui.dropdown(
            options=[task["task"][:60] + "..." for task in debug_tasks],
            label="Select a task to _trace",
            value=debug_tasks[0]["task"][:60] + "...",
        )

        run_traced_demo = mo.ui.run_button(label="🔍 Run Traced Execution")

        cell5_out = mo.vstack(
            [
                mo.md("### 🧪 Traced Agent Testing"),
                mo.md("Select a task to execute with full tracing:"),
                task_selector,
                run_traced_demo,
            ]
        )
    else:
        debug_tasks = None
        task_selector = None
        run_traced_demo = None
        cell5_out = mo.md("")

    output.replace(cell5_out)
    return debug_tasks, run_traced_demo, task_selector


@app.cell
def _(
    available_providers,
    cleandoc,
    debug_tasks,
    execution_tracer,
    mo,
    output,
    run_traced_demo,
    task_selector,
    time,
    traced_agent,
):
    if available_providers and run_traced_demo.value and traced_agent:
        try:
            # Find selected task
            selected_task_data = None
            for task_data in debug_tasks:
                if task_data["task"][:60] in task_selector.value:
                    selected_task_data = task_data
                    break

            if selected_task_data:
                # Execute traced agent
                execution_result = traced_agent.execute(
                    task=selected_task_data["task"],
                    context=selected_task_data["context"],
                )

                if execution_result["success"]:
                    # Get _trace details
                    _trace = execution_tracer.get_trace(execution_result["trace_id"])
                    _result = execution_result["result"]

                    # Format _trace steps
                    step_displays = []
                    for step in _trace["steps"]:
                        timestamp = time.strftime(
                            "%H:%M:%S", time.localtime(step["timestamp"])
                        )
                        step_displays.append(
                            cleandoc(
                                f"""
                                **Step {step['step_number']}: {step['step_name']}** ({timestamp})
                                - Data: {step['step_data']}
                                - Output: {step['step_output'][:100] + '...' if step['step_output'] and len(step['step_output']) > 100 else step['step_output']}
                                """
                            )
                        )

                    cell6_out = mo.vstack(
                        [
                            mo.md("## 🔍 Traced Execution Results"),
                            mo.md(f"**Task:** {selected_task_data['task']}"),
                            mo.md(f"**Context:** {selected_task_data['context']}"),
                            mo.md(f"**Trace ID:** {execution_result['trace_id']}"),
                            mo.md(
                                f"**Execution Time:** {execution_result['execution_time']:.3f}s"
                            ),
                            mo.md("### 🤖 Agent Output"),
                            mo.md(f"**Analysis:** {_result.analysis}"),
                            mo.md(f"**Approach:** {_result.approach}"),
                            mo.md(f"**Execution Steps:** {_result.execution_steps}"),
                            mo.md(f"**Result:** {_result.result}"),
                            mo.md(f"**Confidence:** {_result.confidence}"),
                            mo.md("### 📋 Execution Trace"),
                            mo.md("\n".join(step_displays)),
                            mo.md("### 📊 Performance Metrics"),
                            mo.md(
                                cleandoc(
                                    f"""
                                    - **Total Steps:** {_trace['performance_metrics']['total_steps']}
                                    - **Total Errors:** {_trace['performance_metrics']['total_errors']}
                                    - **Success Rate:** {_trace['performance_metrics']['success_rate']:.2%}
                                    - **Average Step Time:** {_trace['performance_metrics']['avg_step_time']:.3f}s
                                    """
                                )
                            ),
                        ]
                    )
                else:
                    # Show error details
                    _trace = execution_tracer.get_trace(execution_result["trace_id"])

                    cell6_out = mo.vstack(
                        [
                            mo.md("## ❌ Traced Execution Failed"),
                            mo.md(f"**Error:** {execution_result['error']}"),
                            mo.md(f"**Trace ID:** {execution_result['trace_id']}"),
                            mo.md("### 🐛 Error Details"),
                            mo.md(f"**Errors:** {len(_trace['errors'])}"),
                            mo.md(
                                "\n".join(
                                    [
                                        f"- {err['error_type']}: {err['error_message']}"
                                        for err in _trace["errors"]
                                    ]
                                )
                            ),
                        ]
                    )
            else:
                cell6_out = mo.md("❌ Task not found")

        except Exception as e:
            cell6_out = mo.md(f"❌ **Traced Demo Error:** {str(e)}")
    else:
        cell6_out = mo.md(
            "*Select a task and click 'Run Traced Execution' to see detailed tracing*"
        )

    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, execution_tracer, mo, output):
    if available_providers and execution_tracer:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 3: Performance Analysis Dashboard

                Let's analyze the performance of our traced executions:
                """
            )
        )

        # Performance analysis button
        analyze_performance = mo.ui.run_button(label="📊 Analyze Performance")

        cell7_content = mo.vstack(
            [
                mo.md("### 📈 Performance Analysis"),
                mo.md("Analyze performance across all traced executions:"),
                analyze_performance,
            ]
        )
    else:
        cell7_desc = mo.md("")
        analyze_performance = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack(
        [
            cell7_desc,
            cell7_content,
        ]
    )
    output.replace(cell7_out)
    return (analyze_performance,)


@app.cell
def _(
    analyze_performance,
    available_providers,
    cleandoc,
    execution_tracer,
    mo,
    output,
):
    if available_providers and analyze_performance.value and execution_tracer:
        try:
            # Get performance analysis
            performance_data = execution_tracer.analyze_performance()

            if "error" not in performance_data:
                # Get all traces for detailed analysis
                all_traces = execution_tracer.get_all_traces()

                # Create performance summary table
                trace_summary = []
                for _trace in all_traces:
                    trace_summary.append(
                        {
                            "Trace ID": _trace["trace_id"],
                            "Agent": _trace["agent_name"],
                            "Status": _trace["status"],
                            "Steps": _trace["performance_metrics"]["total_steps"],
                            "Errors": _trace["performance_metrics"]["total_errors"],
                            "Time (s)": f"{_trace.get('total_time', 0):.3f}",
                            "Success Rate": f"{_trace['performance_metrics']['success_rate']:.1%}",
                        }
                    )

                # Identify performance issues
                issues = []
                if performance_data["success_rate"] < 0.8:
                    issues.append("Low success rate - investigate error patterns")
                if performance_data["avg_execution_time"] > 10:
                    issues.append("High execution time - optimize slow steps")
                if (
                    performance_data["total_errors"]
                    > performance_data["total_traces"] * 0.5
                ):
                    issues.append("High error rate - review error handling")

                if not issues:
                    issues.append("Performance looks good!")

                cell8_out = mo.vstack(
                    [
                        mo.md("## 📊 Performance Analysis Results"),
                        mo.md("### 🎯 Overall Performance"),
                        mo.md(
                            cleandoc(
                                f"""
                                - **Total Traces:** {performance_data['total_traces']}
                                - **Successful Traces:** {performance_data['successful_traces']}
                                - **Success Rate:** {performance_data['success_rate']:.1%}
                                - **Average Execution Time:** {performance_data['avg_execution_time']:.3f}s
                                - **Average Steps per Trace:** {performance_data['avg_steps_per_trace']:.1f}
                                - **Total Errors:** {performance_data['total_errors']}
                                """
                            )
                        ),
                        mo.md("### 📋 Trace Summary"),
                        mo.ui.table(trace_summary),
                        mo.md("### ⚠️ Performance Issues"),
                        mo.md("\n".join([f"- {issue}" for issue in issues])),
                        mo.md(
                            cleandoc(
                                """
                                ### 💡 Optimization Recommendations

                                **For High Execution Times:**  
                                - Profile individual steps to identify bottlenecks  
                                - Consider parallel execution where possible  
                                - Optimize prompt length and complexity  

                                **For High Error Rates:**  
                                - Review input validation and error handling  
                                - Add retry mechanisms for transient failures  
                                - Improve error recovery strategies  

                                **For Low Success Rates:**  
                                - Analyze failed traces for common patterns  
                                - Improve signature design and field descriptions  
                                - Add more robust validation and fallbacks  
                                """
                            )
                        ),
                    ]
                )
            else:
                # Provide helpful guidance when no traces are available
                if "No traces available" in performance_data["error"]:
                    cell8_out = mo.vstack(
                        [
                            mo.md("## 📊 Performance Analysis"),
                            mo.md("### ⚠️ No Traces Available"),
                            mo.md(
                                cleandoc(
                                    """
                                To analyze performance, you need to run some traced executions first.
                                
                                **Steps to get performance data:**
                                1. Go back to **Step 2** (cell 6) 
                                2. Select a task from the dropdown
                                3. Click **"🔍 Run Traced Execution"**
                                4. Run a few different tasks to get meaningful analysis
                                5. Come back here and click **"📊 Analyze Performance"**
                                
                                **Tip:** Run at least 2-3 different tasks to see interesting performance comparisons!
                                """
                                )
                            ),
                        ]
                    )
                else:
                    cell8_out = mo.md(
                        f"❌ **Performance Analysis Error:** {performance_data['error']}"
                    )

        except Exception as e:
            cell8_out = mo.md(f"❌ **Analysis Error:** {str(e)}")
    else:
        cell8_out = mo.md(
            "*Click 'Analyze Performance' to see performance insights (run some traced executions first)*"
        )

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, execution_tracer, mo, output):
    if available_providers:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 🔧 Step 4: Interactive Debugging Tools

                Let's build interactive tools for real-time debugging:
                """
            )
        )

        # Interactive Debugger
        class InteractiveDebugger:
            """Interactive debugging tools for DSPy agents."""

            def __init__(self, tracer):
                self.tracer = tracer
                self.breakpoints = set()
                self.watch_variables = {}

            def add_breakpoint(self, step_name: str):
                """Add a breakpoint at a specific step."""
                self.breakpoints.add(step_name)
                return f"Breakpoint added at '{step_name}'"

            def remove_breakpoint(self, step_name: str):
                """Remove a breakpoint."""
                self.breakpoints.discard(step_name)
                return f"Breakpoint removed from '{step_name}'"

            def add_watch(self, variable_name: str, expression: str):
                """Add a variable to watch."""
                self.watch_variables[variable_name] = expression
                return f"Watching '{variable_name}': {expression}"

            def inspect_trace(self, trace_id: int, step_number: int = None):
                """Inspect a specific _trace and step."""
                _trace = self.tracer.get_trace(trace_id)
                if not _trace:
                    return {"error": f"Trace {trace_id} not found"}

                if step_number is None:
                    # Return _trace overview
                    return {
                        "trace_id": trace_id,
                        "agent_name": _trace["agent_name"],
                        "status": _trace["status"],
                        "total_steps": len(_trace["steps"]),
                        "total_errors": len(_trace["errors"]),
                        "execution_time": _trace.get("total_time", 0),
                    }
                else:
                    # Return specific step details
                    if step_number <= len(_trace["steps"]):
                        step = _trace["steps"][step_number - 1]
                        return {
                            "step_number": step["step_number"],
                            "step_name": step["step_name"],
                            "timestamp": step["timestamp"],
                            "step_data": step["step_data"],
                            "step_output": step["step_output"],
                        }
                    else:
                        return {"error": f"Step {step_number} not found"}

            def compare_traces(self, trace_id1: int, trace_id2: int):
                """Compare two traces for debugging."""
                trace1 = self.tracer.get_trace(trace_id1)
                trace2 = self.tracer.get_trace(trace_id2)

                if not trace1 or not trace2:
                    return {"error": "One or both traces not found"}

                comparison = {
                    "trace1_steps": len(trace1["steps"]),
                    "trace2_steps": len(trace2["steps"]),
                    "trace1_errors": len(trace1["errors"]),
                    "trace2_errors": len(trace2["errors"]),
                    "trace1_time": trace1.get("total_time", 0),
                    "trace2_time": trace2.get("total_time", 0),
                    "step_differences": [],
                }

                # Compare steps
                max_steps = max(len(trace1["steps"]), len(trace2["steps"]))
                for i in range(max_steps):
                    step1 = trace1["steps"][i] if i < len(trace1["steps"]) else None
                    step2 = trace2["steps"][i] if i < len(trace2["steps"]) else None

                    if step1 and step2:
                        if step1["step_name"] != step2["step_name"]:
                            comparison["step_differences"].append(
                                {
                                    "step_number": i + 1,
                                    "trace1_step": step1["step_name"],
                                    "trace2_step": step2["step_name"],
                                }
                            )
                    elif step1 and not step2:
                        comparison["step_differences"].append(
                            {
                                "step_number": i + 1,
                                "trace1_step": step1["step_name"],
                                "trace2_step": "Missing",
                            }
                        )
                    elif step2 and not step1:
                        comparison["step_differences"].append(
                            {
                                "step_number": i + 1,
                                "trace1_step": "Missing",
                                "trace2_step": step2["step_name"],
                            }
                        )

                return comparison

        # Create interactive debugger
        interactive_debugger = InteractiveDebugger(execution_tracer)

        cell9_content = mo.md(
            cleandoc(
                """
                ### 🔧 Interactive Debugger Created

                **Debugging Features:**  
                - **Breakpoint Management** - Set and remove breakpoints  
                - **Variable Watching** - Monitor specific variables  
                - **Trace Inspection** - Deep dive into execution traces  
                - **Trace Comparison** - Compare different executions  

                The debugger provides powerful interactive debugging capabilities!
                """
            )
        )
    else:
        cell9_desc = mo.md("")
        InteractiveDebugger = None
        interactive_debugger = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack([cell9_desc, cell9_content])
    output.replace(cell9_out)
    return (interactive_debugger,)


@app.cell
def _(available_providers, interactive_debugger, mo, output):
    if available_providers and interactive_debugger:
        # Interactive debugging interface (individual components to avoid _clone issues)
        action_dropdown = mo.ui.dropdown(
            options=[
                "Inspect Trace",
                "Compare Traces",
                "Add Breakpoint",
                "Add Watch",
            ],
            label="Debug Action",
            value="Inspect Trace",
        )
        trace_id_input = mo.ui.number(label="Trace ID", value=1, start=1)
        trace_id2_input = mo.ui.number(
            label="Second Trace ID (for comparison)", value=2, start=1
        )
        step_number_input = mo.ui.number(
            label="Step Number (optional)", value=1, start=1
        )
        breakpoint_name_input = mo.ui.text(
            label="Breakpoint Step Name", placeholder="e.g., prediction_start"
        )
        watch_variable_input = mo.ui.text(
            label="Variable Name to Watch",
            placeholder="e.g., result_confidence",
        )
        watch_expression_input = mo.ui.text(
            label="Watch Expression",
            placeholder="e.g., result.confidence > 0.8",
        )
        execute_debug_button = mo.ui.run_button(label="🔍 Execute Debug Action")

        debug_interface = mo.vstack(
            [
                action_dropdown,
                trace_id_input,
                trace_id2_input,
                step_number_input,
                breakpoint_name_input,
                watch_variable_input,
                watch_expression_input,
                execute_debug_button,
            ]
        )

        cell10_out = mo.vstack(
            [
                mo.md("### 🔧 Interactive Debugging Interface"),
                mo.md("Use the debugging tools to inspect and analyze executions:"),
                debug_interface,
            ]
        )
    else:
        debug_interface = None
        cell10_out = mo.md("")

    output.replace(cell10_out)
    return (
        action_dropdown,
        breakpoint_name_input,
        execute_debug_button,
        step_number_input,
        trace_id2_input,
        trace_id_input,
        watch_expression_input,
        watch_variable_input,
    )


@app.cell
def _(
    action_dropdown,
    available_providers,
    breakpoint_name_input,
    execute_debug_button,
    interactive_debugger,
    json,
    mo,
    output,
    step_number_input,
    trace_id2_input,
    trace_id_input,
    watch_expression_input,
    watch_variable_input,
):
    if available_providers and execute_debug_button.value:
        try:
            action = action_dropdown.value

            if action == "Inspect Trace":
                trace_id = trace_id_input.value
                step_number = (
                    step_number_input.value if step_number_input.value > 0 else None
                )

                _result = interactive_debugger.inspect_trace(trace_id, step_number)

                cell11_out = mo.vstack(
                    [
                        mo.md(f"## 🔍 Trace Inspection: {trace_id}"),
                        mo.md("### 📋 Inspection Result"),
                        mo.md(f"```json\n{json.dumps(_result, indent=2)}\n```"),
                    ]
                )

            elif action == "Compare Traces":
                trace_id1 = trace_id_input.value
                trace_id2 = trace_id2_input.value

                comparison = interactive_debugger.compare_traces(trace_id1, trace_id2)

                cell11_out = mo.vstack(
                    [
                        mo.md(f"## ⚖️ Trace Comparison: {trace_id1} vs {trace_id2}"),
                        mo.md("### 📊 Comparison Results"),
                        mo.md(f"```json\n{json.dumps(comparison, indent=2)}\n```"),
                    ]
                )

            elif action == "Add Breakpoint":
                breakpoint_name = breakpoint_name_input.value
                if breakpoint_name:
                    result = interactive_debugger.add_breakpoint(breakpoint_name)
                    cell11_out = mo.md(f"## 🔴 Breakpoint Added\n\n{result}")
                else:
                    cell11_out = mo.md("❌ Please provide a breakpoint step name")

            elif action == "Add Watch":
                var_name = watch_variable_input.value
                expression = watch_expression_input.value
                if var_name and expression:
                    result = interactive_debugger.add_watch(var_name, expression)
                    cell11_out = mo.md(f"## 👁️ Watch Added\n\n{result}")
                else:
                    cell11_out = mo.md(
                        "❌ Please provide both variable name and expression"
                    )

        except Exception as e:
            cell11_out = mo.md(f"❌ **Debug Action Error:** {str(e)}")
    else:
        cell11_out = mo.md(
            "*Select a debug action and click 'Execute Debug Action' to use the debugging tools*"
        )

    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell12_out = mo.md(
        cleandoc(
            """
                ## 🎓 Advanced Debugging Dashboard Complete!

                ### 🏆 What You've Mastered

                ✅ **Execution Tracing** - Comprehensive step-by-step execution monitoring
                ✅ **Traced Agent Implementation** - Agents with built-in debugging capabilities
                ✅ **Performance Analysis** - Detailed performance metrics and optimization insights
                ✅ **Interactive Debugging** - Real-time debugging tools and interfaces
                ✅ **Error Diagnosis** - Advanced error tracking and analysis

                ### 🛠️ Key Components Built

                1. **DSPyExecutionTracer**
                    - Step-by-step execution logging
                    - Error tracking and analysis
                    - Performance metrics collection
                    - Trace management and retrieval

                2. **TracedAgent**
                    - Agent wrapper with full tracing
                    - Automatic error handling and logging
                    - Performance monitoring integration
                    - State preservation across execution

                3. **Performance Analyzer**
                    - Aggregate performance metrics
                    - Success rate and error analysis
                    - Execution time profiling
                    - Optimization recommendations

                4. **Interactive Debugger**
                    - Breakpoint management
                    - Variable watching
                    - Trace inspection and comparison
                    - Real-time debugging interface

                ### 🎯 Skills Developed

                - **Debugging Architecture** - Designing comprehensive debugging systems
                - **Execution Monitoring** - Tracking and analyzing agent behavior
                - **Performance Profiling** - Identifying and resolving bottlenecks
                - **Error Analysis** - Diagnosing and fixing complex issues
                - **Interactive Tools** - Building user-friendly debugging interfaces

                ### 🚀 Module 02 Complete!

                Congratulations! You've mastered advanced DSPy modules:

                **What You've Built:**
                - ✅ ReAct agents with reasoning and action capabilities
                - ✅ Comprehensive tool integration framework
                - ✅ Multi-step reasoning pipelines
                - ✅ Advanced debugging and monitoring tools

                **Next Module: Retrieval-Augmented Generation (RAG)**
                ```bash
                uv run marimo run 03-retrieval-rag/rag_implementation.py
                ```

                **Coming Up:**
                - Vector database integration
                - Document processing and embedding
                - Retrieval optimization techniques
                - RAG evaluation and monitoring

                ### 💡 Advanced Practice Challenges

                Before moving on, try building:

                1. **Multi-Agent Debugging System**
                    - Debug interactions between multiple agents
                    - Track message passing and coordination
                    - Analyze collective behavior patterns

                2. **Production Monitoring Dashboard**
                    - Real-time agent performance monitoring
                    - Automated alerting for performance issues
                    - Historical trend analysis and reporting

                3. **A/B Testing Framework**
                    - Compare different agent configurations
                    - Statistical significance testing
                    - Automated performance optimization

                4. **Error Recovery System**
                    - Automatic error detection and recovery
                    - Fallback strategy implementation
                    - Self-healing agent capabilities

                ### 🏭 Production Deployment Checklist

                When deploying advanced DSPy systems:
                - [ ] Comprehensive logging and monitoring implemented
                - [ ] Error handling and recovery strategies defined
                - [ ] Performance baselines and alerts configured
                - [ ] Debugging tools accessible to operations team
                - [ ] Documentation for troubleshooting procedures
                - [ ] Regular performance review and optimization schedule

                Master these advanced debugging techniques and you can build, deploy, and maintain sophisticated DSPy systems with confidence!
                """
        )
        if available_providers
        else ""
    )

    output.replace(cell12_out)
    return


if __name__ == "__main__":
    app.run()
