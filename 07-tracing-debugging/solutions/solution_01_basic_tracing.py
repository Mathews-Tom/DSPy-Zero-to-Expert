#!/usr/bin/env python3
"""
Solution 01: Basic DSPy Tracing Implementation

This solution demonstrates how to implement basic tracing for DSPy modules,
including execution tracking, performance measurement, and trace analysis.

Learning Objectives:
- Understand DSPy execution tracing concepts
- Implement basic trace collection
- Analyze execution traces for insights
- Export trace data for further analysis

Author: DSPy Learning Framework
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Represents a single trace event in DSPy execution"""

    event_id: str
    timestamp: float
    event_type: str  # 'start', 'end', 'prediction', 'error'
    module_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Complete execution trace for a DSPy operation"""

    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)
    total_duration: Optional[float] = None
    success: bool = True

    def add_event(self, event: TraceEvent):
        """Add an event to the trace"""
        self.events.append(event)
        if event.event_type == "error":
            self.success = False

    def finalize(self):
        """Finalize the trace with end time and duration"""
        self.end_time = time.time()
        if self.start_time:
            self.total_duration = self.end_time - self.start_time


class BasicTracer:
    """Basic tracing system for DSPy modules"""

    def __init__(self):
        self.current_trace: Optional[ExecutionTrace] = None
        self.traces: List[ExecutionTrace] = []
        self.event_counter = 0

    def start_trace(self, trace_id: str) -> ExecutionTrace:
        """Start a new execution trace"""
        self.current_trace = ExecutionTrace(trace_id=trace_id, start_time=time.time())
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
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
        error: Optional[str] = None,
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
            error=error,
        )

        self.current_trace.add_event(event)
        return event_id

    @contextmanager
    def trace_module(self, module_name: str, inputs: Optional[Dict] = None):
        """Context manager for tracing module execution"""
        start_event_id = self.add_event("start", module_name, inputs=inputs or {})
        start_time = time.time()

        try:
            yield start_event_id
            # Module executed successfully
            duration = time.time() - start_time
            self.add_event("end", module_name)
        except Exception as e:
            # Module execution failed
            duration = time.time() - start_time
            self.add_event("error", module_name, error=str(e))
            raise

    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a trace"""
        trace = next((t for t in self.traces if t.trace_id == trace_id), None)
        if not trace:
            return None

        return {
            "trace_id": trace.trace_id,
            "total_duration": trace.total_duration,
            "success": trace.success,
            "total_events": len(trace.events),
            "event_types": {event.event_type for event in trace.events},
        }

    def export_trace(self, trace_id: str, filepath: str):
        """Export trace to JSON file"""
        trace = next((t for t in self.traces if t.trace_id == trace_id), None)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")

        trace_data = {
            "trace_id": trace.trace_id,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "total_duration": trace.total_duration,
            "success": trace.success,
            "events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "module_name": event.module_name,
                    "inputs": event.inputs,
                    "outputs": event.outputs,
                    "duration": event.duration,
                    "error": event.error,
                }
                for event in trace.events
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)


def create_sample_dspy_module():
    """Create a sample DSPy module for tracing demonstration"""

    class SampleSignature(dspy.Signature):
        """Analyze the given text and provide insights"""

        text = dspy.InputField(desc="Text to analyze")
        analysis = dspy.OutputField(desc="Analysis of the text")

    # For demonstration, we'll use a mock module since we don't have LLM configured
    class MockPredict:
        def __init__(self, signature):
            self.signature = signature

        def __call__(self, **kwargs):
            # Simulate processing time
            time.sleep(0.1)

            # Mock response
            class MockResult:
                def __init__(self, analysis):
                    self.analysis = analysis

            return MockResult(f"Analysis of: {kwargs.get('text', 'No text provided')}")

    return MockPredict(SampleSignature)


def demonstrate_basic_tracing():
    """Demonstrate basic tracing functionality"""
    print("=== Basic DSPy Tracing Demonstration ===\n")

    # Initialize tracer
    tracer = BasicTracer()

    # Create sample module
    module = create_sample_dspy_module()

    # Example 1: Successful execution with tracing
    print("1. Tracing successful execution:")
    trace = tracer.start_trace("successful_execution")

    with tracer.trace_module(
        "MockPredict", {"text": "Machine learning is fascinating"}
    ):
        tracer.add_event(
            "prediction",
            "MockPredict",
            inputs={"text": "Machine learning is fascinating"},
        )
        result = module(text="Machine learning is fascinating")
        tracer.add_event(
            "prediction", "MockPredict", outputs={"analysis": result.analysis}
        )

    tracer.end_trace()

    # Print trace summary
    summary = tracer.get_trace_summary("successful_execution")
    print(f"Trace Summary: {json.dumps(summary, indent=2)}\n")

    # Example 2: Execution with error
    print("2. Tracing execution with error:")
    trace = tracer.start_trace("error_execution")

    try:
        with tracer.trace_module("MockPredict", {"text": None}):
            # Simulate an error
            raise ValueError("Invalid input: text cannot be None")
    except ValueError as e:
        print(f"Caught error: {e}")

    tracer.end_trace()

    # Print trace summary
    summary = tracer.get_trace_summary("error_execution")
    print(f"Error Trace Summary: {json.dumps(summary, indent=2)}\n")

    # Example 3: Export trace data
    print("3. Exporting trace data:")
    try:
        tracer.export_trace("successful_execution", "successful_trace.json")
        print("✅ Trace exported to successful_trace.json")

        tracer.export_trace("error_execution", "error_trace.json")
        print("✅ Error trace exported to error_trace.json")
    except Exception as e:
        print(f"❌ Export failed: {e}")

    # Example 4: Analyze all traces
    print("\n4. Analyzing all traces:")
    print(f"Total traces collected: {len(tracer.traces)}")

    for trace in tracer.traces:
        print(
            f"- {trace.trace_id}: {trace.total_duration:.3f}s, "
            f"Success: {trace.success}, Events: {len(trace.events)}"
        )


def analyze_trace_performance():
    """Demonstrate trace performance analysis"""
    print("\n=== Trace Performance Analysis ===\n")

    tracer = BasicTracer()
    module = create_sample_dspy_module()

    # Run multiple executions to collect performance data
    execution_times = []

    for i in range(5):
        trace_id = f"performance_test_{i+1}"
        tracer.start_trace(trace_id)

        start_time = time.time()
        with tracer.trace_module("MockPredict", {"text": f"Test input {i+1}"}):
            result = module(text=f"Test input {i+1}")

        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        tracer.end_trace()

    # Analyze performance
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    print(f"Performance Analysis:")
    print(f"- Average execution time: {avg_time:.3f}s")
    print(f"- Minimum execution time: {min_time:.3f}s")
    print(f"- Maximum execution time: {max_time:.3f}s")
    print(f"- Total executions: {len(execution_times)}")

    # Identify slow executions
    slow_threshold = avg_time * 1.5
    slow_executions = [i for i, t in enumerate(execution_times) if t > slow_threshold]

    if slow_executions:
        print(f"- Slow executions (>{slow_threshold:.3f}s): {slow_executions}")
    else:
        print("- No slow executions detected")


def demonstrate_trace_filtering():
    """Demonstrate trace filtering and analysis"""
    print("\n=== Trace Filtering and Analysis ===\n")

    tracer = BasicTracer()
    module = create_sample_dspy_module()

    # Create traces with different characteristics
    test_cases = [
        ("quick_execution", "Short text", 0.05),
        ("medium_execution", "Medium length text for analysis", 0.1),
        ("slow_execution", "Very long text that takes more time to process", 0.2),
    ]

    for trace_id, text, delay in test_cases:
        tracer.start_trace(trace_id)

        with tracer.trace_module("MockPredict", {"text": text}):
            time.sleep(delay)  # Simulate different processing times
            result = module(text=text)

        tracer.end_trace()

    # Filter and analyze traces
    print("Trace Analysis:")

    # Find fastest and slowest traces
    fastest_trace = min(tracer.traces, key=lambda t: t.total_duration)
    slowest_trace = max(tracer.traces, key=lambda t: t.total_duration)

    print(
        f"- Fastest trace: {fastest_trace.trace_id} ({fastest_trace.total_duration:.3f}s)"
    )
    print(
        f"- Slowest trace: {slowest_trace.trace_id} ({slowest_trace.total_duration:.3f}s)"
    )

    # Filter traces by duration
    slow_threshold = 0.15
    slow_traces = [t for t in tracer.traces if t.total_duration > slow_threshold]

    print(f"- Traces slower than {slow_threshold}s: {len(slow_traces)}")
    for trace in slow_traces:
        print(f"  - {trace.trace_id}: {trace.total_duration:.3f}s")


if __name__ == "__main__":
    """
    Exercise Solution: Basic DSPy Tracing

    This script demonstrates:
    1. Basic trace collection and management
    2. Context manager for automatic tracing
    3. Error handling in traces
    4. Trace export and analysis
    5. Performance analysis using traces
    6. Trace filtering and insights
    """

    try:
        demonstrate_basic_tracing()
        analyze_trace_performance()
        demonstrate_trace_filtering()

        print("\n✅ Basic tracing exercise completed successfully!")
        print("\nKey takeaways:")
        print("- Tracing provides visibility into DSPy module execution")
        print("- Context managers simplify trace management")
        print("- Trace data can be exported for further analysis")
        print("- Performance patterns can be identified through trace analysis")

    except Exception as e:
        print(f"\n❌ Exercise failed: {e}")
        logger.exception("Exercise execution failed")
