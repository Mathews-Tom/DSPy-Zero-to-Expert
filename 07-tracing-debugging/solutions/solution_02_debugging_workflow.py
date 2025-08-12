#!/usr/bin/env python3
"""
Solution 02: DSPy Debugging Workflow Implementation

This solution demonstrates a comprehensive debugging workflow for DSPy modules,
including breakpoints, state inspection, error analysis, and debugging strategies.

Learning Objectives:
- Implement interactive debugging for DSPy modules
- Use breakpoints and state inspection effectively
- Analyze and diagnose common DSPy errors
- Develop systematic debugging strategies

Author: DSPy Learning Framework
"""

import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dspy

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class DebugPoint:
    """Represents a debugging breakpoint"""

    point_id: str
    module_name: str
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0


@dataclass
class ExecutionState:
    """Captures the execution state at a debug point"""

    point_id: str
    timestamp: float
    module_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    local_vars: Dict[str, Any] = field(default_factory=dict)
    stack_trace: List[str] = field(default_factory=list)


class DSPyDebugger:
    """Comprehensive debugging system for DSPy modules"""

    def __init__(self):
        self.breakpoints: Dict[str, DebugPoint] = {}
        self.execution_states: List[ExecutionState] = []
        self.errors: List[Dict[str, Any]] = []
        self.debug_mode = False

    def add_breakpoint(
        self, point_id: str, module_name: str, condition: Optional[str] = None
    ):
        """Add a debugging breakpoint"""
        breakpoint = DebugPoint(
            point_id=point_id, module_name=module_name, condition=condition
        )
        self.breakpoints[point_id] = breakpoint
        logger.info(f"Added breakpoint: {point_id} for {module_name}")
        return breakpoint

    def remove_breakpoint(self, point_id: str):
        """Remove a breakpoint"""
        if point_id in self.breakpoints:
            del self.breakpoints[point_id]
            logger.info(f"Removed breakpoint: {point_id}")

    def should_break(self, point_id: str, context: Dict[str, Any]) -> bool:
        """Check if execution should break at this point"""
        if point_id not in self.breakpoints:
            return False

        breakpoint = self.breakpoints[point_id]
        if not breakpoint.enabled:
            return False

        # Check condition if specified
        if breakpoint.condition:
            try:
                # Simple condition evaluation
                return eval(breakpoint.condition, {"__builtins__": {}}, context)
            except Exception as e:
                logger.warning(f"Breakpoint condition error: {e}")
                return False

        return True

    def capture_state(
        self,
        point_id: str,
        module_name: str,
        inputs: Optional[Dict] = None,
        outputs: Optional[Dict] = None,
        local_vars: Optional[Dict] = None,
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

        self.execution_states.append(state)

        # Update breakpoint hit count
        if point_id in self.breakpoints:
            self.breakpoints[point_id].hit_count += 1

        return state

    def analyze_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            error_info["suggestions"].extend(
                [
                    "Check if the object has the expected attributes",
                    "Verify that the module was properly initialized",
                    "Ensure all required fields are present in the signature",
                ]
            )
        elif isinstance(error, KeyError):
            error_info["suggestions"].extend(
                [
                    "Check if all required keys are present in the input",
                    "Verify the signature field names match the input data",
                    "Ensure the DSPy module signature is correctly defined",
                ]
            )
        elif isinstance(error, ValueError):
            error_info["suggestions"].extend(
                [
                    "Check input data types and formats",
                    "Verify that input values are within expected ranges",
                    "Ensure all required fields have valid values",
                ]
            )
        elif "API" in str(error) or "rate limit" in str(error).lower():
            error_info["suggestions"].extend(
                [
                    "Check API key configuration and rate limits",
                    "Consider adding retry logic or reducing request frequency",
                    "Verify network connectivity and API endpoint",
                ]
            )

        self.errors.append(error_info)
        return error_info

    def get_variable_info(self, var_name: str, value: Any) -> Dict[str, Any]:
        """Get detailed information about a variable"""
        return {
            "name": var_name,
            "type": type(value).__name__,
            "value": str(value)[:500],  # Truncate long values
            "size": len(str(value)),
            "is_callable": callable(value),
            "attributes": [attr for attr in dir(value) if not attr.startswith("_")],
        }

    def debug_module_execution(
        self, module, inputs: Dict[str, Any], breakpoint_id: str = "execution"
    ):
        """Debug a module execution with automatic state capture"""
        logger.info(f"Starting debug execution for {type(module).__name__}")

        # Capture initial state
        if self.should_break(f"{breakpoint_id}_start", inputs):
            self.capture_state(
                f"{breakpoint_id}_start",
                type(module).__name__,
                inputs=inputs,
                local_vars={"debug_mode": True},
            )
            print(f"🔴 Breakpoint hit: {breakpoint_id}_start")
            print(f"Inputs: {json.dumps(inputs, indent=2)}")

        try:
            # Execute module
            result = module(**inputs)

            # Capture success state
            outputs = (
                {"result": str(result)}
                if hasattr(result, "__dict__")
                else {"result": str(result)}
            )
            if self.should_break(
                f"{breakpoint_id}_success", {"inputs": inputs, "outputs": outputs}
            ):
                self.capture_state(
                    f"{breakpoint_id}_success",
                    type(module).__name__,
                    inputs=inputs,
                    outputs=outputs,
                )
                print(f"🟢 Breakpoint hit: {breakpoint_id}_success")
                print(f"Outputs: {json.dumps(outputs, indent=2)}")

            return result

        except Exception as e:
            # Analyze and capture error state
            error_analysis = self.analyze_error(e, inputs)
            self.capture_state(
                f"{breakpoint_id}_error",
                type(module).__name__,
                inputs=inputs,
                local_vars={"error": str(e)},
            )

            print(f"🔴 Error breakpoint hit: {breakpoint_id}_error")
            print(f"Error: {error_analysis['type']}: {error_analysis['message']}")
            print("Suggestions:")
            for suggestion in error_analysis["suggestions"]:
                print(f"  - {suggestion}")

            raise


def create_problematic_dspy_modules():
    """Create DSPy modules with common issues for debugging demonstration"""

    # Module 1: Signature mismatch issue
    class ProblematicSignature1(dspy.Signature):
        """Signature with field name mismatch"""

        input_text = dspy.InputField(desc="Input text")
        output_result = dspy.OutputField(desc="Output result")

    class MockProblematicModule1:
        def __init__(self):
            self.signature = ProblematicSignature1

        def __call__(self, **kwargs):
            # This will cause KeyError if wrong field name is used
            if "wrong_field" in kwargs:
                raise KeyError("Field 'wrong_field' not found in signature")

            text = kwargs.get("input_text", "")
            if not text:
                raise ValueError("Input text cannot be empty")

            class Result:
                def __init__(self, output_result):
                    self.output_result = output_result

            return Result(f"Processed: {text}")

    # Module 2: Attribute access issue
    class MockProblematicModule2:
        def __init__(self):
            self.initialized = False

        def __call__(self, **kwargs):
            if not self.initialized:
                raise AttributeError("Module not properly initialized")

            return {"result": "Success"}

    # Module 3: Value validation issue
    class MockProblematicModule3:
        def __call__(self, **kwargs):
            value = kwargs.get("numeric_input", 0)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected numeric input, got {type(value).__name__}")

            if value < 0:
                raise ValueError("Numeric input must be non-negative")

            return {"result": f"Square root: {value ** 0.5}"}

    return {
        "signature_mismatch": MockProblematicModule1(),
        "attribute_error": MockProblematicModule2(),
        "value_validation": MockProblematicModule3(),
    }


def demonstrate_debugging_workflow():
    """Demonstrate comprehensive debugging workflow"""
    print("=== DSPy Debugging Workflow Demonstration ===\n")

    debugger = DSPyDebugger()
    modules = create_problematic_dspy_modules()

    # Set up breakpoints
    debugger.add_breakpoint("execution_start", "MockProblematicModule", condition=None)
    debugger.add_breakpoint(
        "execution_success", "MockProblematicModule", condition=None
    )
    debugger.add_breakpoint("execution_error", "MockProblematicModule", condition=None)

    print("1. Debugging signature mismatch issue:")
    print("-" * 40)

    try:
        # This should trigger a KeyError
        result = debugger.debug_module_execution(
            modules["signature_mismatch"],
            {"wrong_field": "test input"},  # Wrong field name
            "signature_test",
        )
    except Exception as e:
        print(f"Caught and analyzed error: {type(e).__name__}")

    print("\n2. Debugging attribute error issue:")
    print("-" * 40)

    try:
        # This should trigger an AttributeError
        result = debugger.debug_module_execution(
            modules["attribute_error"], {"input": "test"}, "attribute_test"
        )
    except Exception as e:
        print(f"Caught and analyzed error: {type(e).__name__}")

    print("\n3. Debugging value validation issue:")
    print("-" * 40)

    try:
        # This should trigger a ValueError
        result = debugger.debug_module_execution(
            modules["value_validation"],
            {"numeric_input": "not_a_number"},
            "validation_test",
        )
    except Exception as e:
        print(f"Caught and analyzed error: {type(e).__name__}")

    print("\n4. Successful execution after fixing issues:")
    print("-" * 40)

    try:
        # Fix the issues and run successfully
        modules["attribute_error"].initialized = True  # Fix initialization

        result1 = debugger.debug_module_execution(
            modules["signature_mismatch"],
            {"input_text": "correct field name"},  # Correct field name
            "fixed_signature",
        )
        print(f"✅ Fixed signature result: {result1.output_result}")

        result2 = debugger.debug_module_execution(
            modules["attribute_error"], {"input": "test"}, "fixed_attribute"
        )
        print(f"✅ Fixed attribute result: {result2}")

        result3 = debugger.debug_module_execution(
            modules["value_validation"],
            {"numeric_input": 16},  # Correct numeric input
            "fixed_validation",
        )
        print(f"✅ Fixed validation result: {result3}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def demonstrate_state_inspection():
    """Demonstrate state inspection and variable analysis"""
    print("\n=== State Inspection and Variable Analysis ===\n")

    debugger = DSPyDebugger()

    # Create a module for state inspection
    class InspectionModule:
        def __init__(self):
            self.config = {"max_length": 100, "temperature": 0.7}
            self.history = []

        def __call__(self, **kwargs):
            text = kwargs.get("text", "")
            self.history.append(text)

            # Simulate processing
            processed = text.upper()[: self.config["max_length"]]

            return {"processed": processed, "length": len(processed)}

    module = InspectionModule()

    # Add conditional breakpoint
    debugger.add_breakpoint(
        "long_input", "InspectionModule", condition="len(inputs.get('text', '')) > 50"
    )

    test_inputs = [
        {"text": "Short text"},
        {
            "text": "This is a much longer text that should trigger the conditional breakpoint because it exceeds 50 characters"
        },
        {"text": "Medium length text for testing"},
    ]

    for i, inputs in enumerate(test_inputs):
        print(f"Test {i+1}: Processing text of length {len(inputs['text'])}")

        # Check if breakpoint should trigger
        if debugger.should_break("long_input", {"inputs": inputs}):
            print("🔴 Conditional breakpoint triggered!")

            # Capture detailed state
            state = debugger.capture_state(
                "long_input",
                "InspectionModule",
                inputs=inputs,
                local_vars={
                    "module_config": module.config,
                    "module_history": module.history,
                    "text_length": len(inputs["text"]),
                },
            )

            # Inspect variables
            for var_name, value in state.local_vars.items():
                var_info = debugger.get_variable_info(var_name, value)
                print(f"  Variable '{var_name}':")
                print(f"    Type: {var_info['type']}")
                print(f"    Value: {var_info['value']}")
                print(f"    Size: {var_info['size']}")

        # Execute module
        result = module(**inputs)
        print(f"  Result: {result}\n")


def demonstrate_error_pattern_analysis():
    """Demonstrate error pattern analysis and debugging strategies"""
    print("\n=== Error Pattern Analysis ===\n")

    debugger = DSPyDebugger()

    # Simulate various error scenarios
    error_scenarios = [
        {
            "name": "API Rate Limit",
            "error": Exception("Rate limit exceeded. Please try again later."),
            "context": {"api_calls": 1000, "time_window": "1 hour"},
        },
        {
            "name": "Invalid Signature Field",
            "error": KeyError("'output_field' not found in signature"),
            "context": {
                "signature_fields": ["input", "result"],
                "accessed_field": "output_field",
            },
        },
        {
            "name": "Model Initialization",
            "error": AttributeError("'NoneType' object has no attribute 'generate'"),
            "context": {"model_configured": False, "api_key_set": True},
        },
        {
            "name": "Input Validation",
            "error": ValueError("Input text must be a string, got int"),
            "context": {"input_type": "int", "expected_type": "str"},
        },
    ]

    print("Analyzing common error patterns:\n")

    for scenario in error_scenarios:
        print(f"Scenario: {scenario['name']}")
        print("-" * 30)

        error_analysis = debugger.analyze_error(scenario["error"], scenario["context"])

        print(f"Error Type: {error_analysis['type']}")
        print(f"Message: {error_analysis['message']}")
        print("Debugging Suggestions:")
        for suggestion in error_analysis["suggestions"]:
            print(f"  • {suggestion}")
        print()

    # Analyze error patterns
    print("Error Pattern Summary:")
    print("-" * 30)

    error_types = {}
    for error in debugger.errors:
        error_type = error["type"]
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1

    for error_type, count in error_types.items():
        print(f"  {error_type}: {count} occurrence(s)")


def demonstrate_debugging_best_practices():
    """Demonstrate debugging best practices and strategies"""
    print("\n=== Debugging Best Practices ===\n")

    print("1. Systematic Debugging Approach:")
    print("   • Start with the error message and stack trace")
    print("   • Identify the failing component (signature, module, input)")
    print("   • Use breakpoints to isolate the issue")
    print("   • Inspect variable states at critical points")
    print("   • Test fixes incrementally")

    print("\n2. Common DSPy Debugging Patterns:")
    print("   • Signature field mismatches → Check field names and types")
    print("   • Module initialization issues → Verify configuration and setup")
    print("   • API-related errors → Check keys, limits, and connectivity")
    print("   • Input validation errors → Validate data types and formats")

    print("\n3. Debugging Tools and Techniques:")
    print("   • Use conditional breakpoints for specific scenarios")
    print("   • Capture and analyze execution states")
    print("   • Log intermediate results and transformations")
    print("   • Test with minimal reproducible examples")

    print("\n4. Prevention Strategies:")
    print("   • Implement comprehensive input validation")
    print("   • Use type hints and documentation")
    print("   • Add error handling and recovery mechanisms")
    print("   • Write unit tests for edge cases")


if __name__ == "__main__":
    """
    Exercise Solution: DSPy Debugging Workflow

    This script demonstrates:
    1. Comprehensive debugging workflow for DSPy modules
    2. Breakpoint management and conditional debugging
    3. State inspection and variable analysis
    4. Error pattern analysis and suggestions
    5. Debugging best practices and strategies
    """

    try:
        demonstrate_debugging_workflow()
        demonstrate_state_inspection()
        demonstrate_error_pattern_analysis()
        demonstrate_debugging_best_practices()

        print("\n✅ Debugging workflow exercise completed successfully!")
        print("\nKey takeaways:")
        print("- Systematic debugging approach improves efficiency")
        print("- Breakpoints and state inspection provide deep insights")
        print("- Error analysis helps identify common patterns")
        print("- Prevention strategies reduce debugging needs")

    except Exception as e:
        print(f"\n❌ Exercise failed: {e}")
        logger.exception("Exercise execution failed")
