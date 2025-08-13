#!/usr/bin/env python3
"""
DSPy Module Composition and Workflow System

This module provides comprehensive tools for combining custom DSPy modules into complex
workflows, including orchestration, configuration management, and serialization capabilities.

Learning Objectives:
- Understand module composition patterns and strategies
- Learn to create complex workflows from simple modules
- Master configuration management and parameter tuning
- Implement module serialization and sharing capabilities

Author: DSPy Learning Framework
"""

import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dspy
import yaml
from dspy import Example

# Import from our custom module framework
try:
    from .component_library import (
        ComponentPipeline,
        ComponentRouter,
        ComponentType,
        ReusableComponent,
    )
    from .custom_module_template import (
        CustomModuleBase,
        ModuleMetadata,
        ModuleValidator,
    )
    from .module_testing_framework import ModuleTestRunner, TestCase, TestSuite
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from component_library import (
        ComponentPipeline,
        ComponentRouter,
        ComponentType,
        ReusableComponent,
    )
    from custom_module_template import CustomModuleBase, ModuleMetadata, ModuleValidator
    from module_testing_framework import ModuleTestRunner, TestCase, TestSuite

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowExecutionMode(Enum):
    """Execution modes for workflows"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    STREAMING = "streaming"


class WorkflowStatus(Enum):
    """Status of workflow execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ModuleConfiguration:
    """Configuration for a module in a workflow"""

    module_id: str
    module_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    input_mappings: dict[str, str] = field(default_factory=dict)
    output_mappings: dict[str, str] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    retry_config: dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None


@dataclass
class WorkflowConfiguration:
    """Configuration for an entire workflow"""

    workflow_id: str
    name: str
    description: str
    version: str = "1.0.0"
    execution_mode: WorkflowExecutionMode = WorkflowExecutionMode.SEQUENTIAL
    modules: list[ModuleConfiguration] = field(default_factory=list)
    global_parameters: dict[str, Any] = field(default_factory=dict)
    error_handling: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for workflow execution"""

    workflow_id: str
    execution_id: str
    start_time: float
    current_step: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    variables: dict[str, Any] = field(default_factory=dict)
    step_results: list[dict[str, Any]] = field(default_factory=list)
    error_log: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)


class ModuleComposer:
    """Composer for creating complex module compositions"""

    def __init__(self):
        self.registered_modules = {}
        self.composition_patterns = {}

    def register_module(self, module_id: str, module: CustomModuleBase):
        """Register a module for composition"""
        self.registered_modules[module_id] = module
        logger.info("Registered module: %s", module_id)

    def create_sequential_composition(
        self, module_ids: list[str], name: str = "Sequential Composition"
    ) -> "SequentialComposition":
        """Create a sequential composition of modules"""
        modules = [
            self.registered_modules[mid]
            for mid in module_ids
            if mid in self.registered_modules
        ]
        return SequentialComposition(modules, name)

    def create_parallel_composition(
        self, module_ids: list[str], name: str = "Parallel Composition"
    ) -> "ParallelComposition":
        """Create a parallel composition of modules"""
        modules = [
            self.registered_modules[mid]
            for mid in module_ids
            if mid in self.registered_modules
        ]
        return ParallelComposition(modules, name)

    def create_conditional_composition(
        self,
        condition_func: Callable,
        true_module_id: str,
        false_module_id: str,
        name: str = "Conditional Composition",
    ) -> "ConditionalComposition":
        """Create a conditional composition of modules"""
        true_module = self.registered_modules.get(true_module_id)
        false_module = self.registered_modules.get(false_module_id)

        if not true_module or not false_module:
            raise ValueError("Both true and false modules must be registered")

        return ConditionalComposition(condition_func, true_module, false_module, name)

    def create_feedback_composition(
        self,
        module_ids: list[str],
        feedback_condition: Callable,
        max_iterations: int = 10,
        name: str = "Feedback Composition",
    ) -> "FeedbackComposition":
        """Create a feedback composition with iterative processing"""
        modules = [
            self.registered_modules[mid]
            for mid in module_ids
            if mid in self.registered_modules
        ]
        return FeedbackComposition(modules, feedback_condition, max_iterations, name)


class BaseComposition(CustomModuleBase, ABC):
    """Base class for all module compositions"""

    def __init__(self, modules: List[CustomModuleBase], name: str):
        metadata = ModuleMetadata(
            name=name,
            description=f"Composition of {len(modules)} modules",
            version="1.0.0",
            author="DSPy Module Composition",
        )
        super().__init__(metadata)
        self.modules = modules
        self.composition_id = str(uuid.uuid4())
        self._initialized = True

    @abstractmethod
    def forward(self, **kwargs) -> Any:
        """Execute the composition"""
        pass

    def get_composition_info(self) -> dict[str, Any]:
        """Get information about the composition"""
        return {
            "composition_id": self.composition_id,
            "name": self.metadata.name,
            "module_count": len(self.modules),
            "modules": [
                {
                    "class": module.__class__.__name__,
                    "metadata": (
                        module.get_info() if hasattr(module, "get_info") else {}
                    ),
                }
                for module in self.modules
            ],
        }


class SequentialComposition(BaseComposition):
    """Sequential execution of modules"""

    def forward(self, **kwargs) -> dict[str, Any]:
        """Execute modules sequentially"""
        results = []
        current_input = kwargs.copy()

        for i, module in enumerate(self.modules):
            try:
                start_time = time.time()
                result = module(**current_input)
                execution_time = time.time() - start_time

                step_result = {
                    "step": i,
                    "module": module.__class__.__name__,
                    "result": result,
                    "execution_time": execution_time,
                    "success": True,
                }
                results.append(step_result)

                # Update input for next module
                if isinstance(result, dict):
                    current_input.update(result)
                else:
                    current_input["previous_result"] = result

            except Exception as e:
                step_result = {
                    "step": i,
                    "module": module.__class__.__name__,
                    "error": str(e),
                    "success": False,
                }
                results.append(step_result)
                logger.error(
                    "Sequential composition failed at step %s: %s", i, e, exc_info=True
                )
                break

        return {
            "composition_type": "sequential",
            "composition_id": self.composition_id,
            "steps": results,
            "total_steps": len(self.modules),
            "successful_steps": sum(1 for r in results if r.get("success", False)),
            "final_result": (
                results[-1]["result"]
                if results and results[-1].get("success")
                else None
            ),
        }


class ParallelComposition(BaseComposition):
    """Parallel execution of modules"""

    def forward(self, **kwargs) -> dict[str, Any]:
        """Execute modules in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        def execute_module(module, module_index):
            try:
                start_time = time.time()
                result = module(**kwargs)
                execution_time = time.time() - start_time

                return {
                    "step": module_index,
                    "module": module.__class__.__name__,
                    "result": result,
                    "execution_time": execution_time,
                    "success": True,
                }
            except Exception as e:
                return {
                    "step": module_index,
                    "module": module.__class__.__name__,
                    "error": str(e),
                    "success": False,
                }

        with ThreadPoolExecutor(max_workers=len(self.modules)) as executor:
            future_to_module = {
                executor.submit(execute_module, module, i): (module, i)
                for i, module in enumerate(self.modules)
            }

            for future in as_completed(future_to_module):
                result = future.result()
                results.append(result)

        # Sort results by step order
        results.sort(key=lambda x: x["step"])

        # Aggregate successful results
        successful_results = [
            r["result"] for r in results if r.get("success") and "result" in r
        ]

        return {
            "composition_type": "parallel",
            "composition_id": self.composition_id,
            "steps": results,
            "total_steps": len(self.modules),
            "successful_steps": sum(1 for r in results if r.get("success", False)),
            "aggregated_results": successful_results,
            "parallel_execution_time": max(r.get("execution_time", 0) for r in results),
        }


class ConditionalComposition(BaseComposition):
    """Conditional execution of modules"""

    def __init__(
        self,
        condition_func: Callable,
        true_module: CustomModuleBase,
        false_module: CustomModuleBase,
        name: str,
    ):
        super().__init__([true_module, false_module], name)
        self.condition_func = condition_func
        self.true_module = true_module
        self.false_module = false_module

    def forward(self, **kwargs) -> dict[str, Any]:
        """Execute module based on condition"""
        try:
            condition_result = self.condition_func(**kwargs)
            selected_module = (
                self.true_module if condition_result else self.false_module
            )

            start_time = time.time()
            result = selected_module(**kwargs)
            execution_time = time.time() - start_time

            return {
                "composition_type": "conditional",
                "composition_id": self.composition_id,
                "condition_result": condition_result,
                "selected_module": selected_module.__class__.__name__,
                "result": result,
                "execution_time": execution_time,
                "success": True,
            }

        except Exception as e:
            return {
                "composition_type": "conditional",
                "composition_id": self.composition_id,
                "error": str(e),
                "success": False,
            }


class FeedbackComposition(BaseComposition):
    """Feedback composition with iterative processing"""

    def __init__(
        self,
        modules: List[CustomModuleBase],
        feedback_condition: Callable,
        max_iterations: int,
        name: str,
    ):
        super().__init__(modules, name)
        self.feedback_condition = feedback_condition
        self.max_iterations = max_iterations

    def forward(self, **kwargs) -> dict[str, Any]:
        """Execute modules with feedback loop"""
        iterations = []
        current_input = kwargs.copy()

        for iteration in range(self.max_iterations):
            iteration_results = []

            # Execute all modules in sequence
            for i, module in enumerate(self.modules):
                try:
                    start_time = time.time()
                    result = module(**current_input)
                    execution_time = time.time() - start_time

                    step_result = {
                        "step": i,
                        "module": module.__class__.__name__,
                        "result": result,
                        "execution_time": execution_time,
                        "success": True,
                    }
                    iteration_results.append(step_result)

                    # Update input for next module
                    if isinstance(result, dict):
                        current_input.update(result)
                    else:
                        current_input["previous_result"] = result

                except Exception as e:
                    step_result = {
                        "step": i,
                        "module": module.__class__.__name__,
                        "error": str(e),
                        "success": False,
                    }
                    iteration_results.append(step_result)
                    break

            iterations.append(
                {
                    "iteration": iteration,
                    "steps": iteration_results,
                    "successful_steps": sum(
                        1 for r in iteration_results if r.get("success", False)
                    ),
                }
            )

            # Check feedback condition
            try:
                if self.feedback_condition(current_input, iteration_results):
                    logger.info(
                        f"Feedback condition satisfied after {iteration + 1} iterations"
                    )
                    break
            except Exception as e:
                logger.warning(
                    "Feedback condition evaluation failed: %s", e, exc_info=True
                )
                break

        return {
            "composition_type": "feedback",
            "composition_id": self.composition_id,
            "total_iterations": len(iterations),
            "max_iterations": self.max_iterations,
            "iterations": iterations,
            "final_result": current_input,
            "converged": len(iterations) < self.max_iterations,
        }


class WorkflowOrchestrator:
    """Orchestrator for managing complex workflows"""

    def __init__(self):
        self.workflows = {}
        self.execution_contexts = {}
        self.module_registry = {}

    def register_workflow(self, config: WorkflowConfiguration):
        """Register a workflow configuration"""
        self.workflows[config.workflow_id] = config
        logger.info("Registered workflow: %s (%s)", config.name, config.workflow_id)

    def register_module(self, module_id: str, module: CustomModuleBase):
        """Register a module for use in workflows"""
        self.module_registry[module_id] = module
        logger.info("Registered module for workflows: %s", module_id)

    def execute_workflow(
        self, workflow_id: str, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a registered workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        config = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())

        # Create execution context
        context = ExecutionContext(
            workflow_id=workflow_id,
            execution_id=execution_id,
            start_time=time.time(),
            variables=inputs.copy(),
        )

        self.execution_contexts[execution_id] = context

        try:
            context.status = WorkflowStatus.RUNNING

            if config.execution_mode == WorkflowExecutionMode.SEQUENTIAL:
                result = self._execute_sequential_workflow(config, context)
            elif config.execution_mode == WorkflowExecutionMode.PARALLEL:
                result = self._execute_parallel_workflow(config, context)
            elif config.execution_mode == WorkflowExecutionMode.CONDITIONAL:
                result = self._execute_conditional_workflow(config, context)
            else:
                raise ValueError(f"Unsupported execution mode: {config.execution_mode}")

            context.status = WorkflowStatus.COMPLETED

        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.error_log.append(
                {
                    "timestamp": time.time(),
                    "error": str(e),
                    "step": context.current_step,
                }
            )
            logger.error(f"Workflow execution failed: {e}")
            result = {"error": str(e), "execution_id": execution_id}

        # Calculate performance metrics
        context.performance_metrics = {
            "total_execution_time": time.time() - context.start_time,
            "steps_completed": context.current_step,
            "success_rate": len(
                [r for r in context.step_results if r.get("success", False)]
            )
            / max(len(context.step_results), 1),
        }

        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": context.status.value,
            "result": result,
            "performance_metrics": context.performance_metrics,
            "execution_context": context,
        }

    def _execute_sequential_workflow(
        self, config: WorkflowConfiguration, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute workflow sequentially"""
        results = []

        for module_config in config.modules:
            context.current_step += 1

            # Get module
            if module_config.module_type not in self.module_registry:
                raise ValueError(f"Module not registered: {module_config.module_type}")

            module = self.module_registry[module_config.module_type]

            # Prepare inputs
            module_inputs = self._prepare_module_inputs(module_config, context)

            # Execute module
            try:
                start_time = time.time()
                result = module(**module_inputs)
                execution_time = time.time() - start_time

                step_result = {
                    "step": context.current_step,
                    "module_id": module_config.module_id,
                    "module_type": module_config.module_type,
                    "result": result,
                    "execution_time": execution_time,
                    "success": True,
                }

                # Update context variables
                self._update_context_variables(module_config, result, context)

            except Exception as e:
                step_result = {
                    "step": context.current_step,
                    "module_id": module_config.module_id,
                    "module_type": module_config.module_type,
                    "error": str(e),
                    "success": False,
                }

                # Handle error based on configuration
                if config.error_handling.get("stop_on_error", True):
                    raise

            results.append(step_result)
            context.step_results.append(step_result)

        return {
            "execution_mode": "sequential",
            "steps": results,
            "final_variables": context.variables,
        }

    def _execute_parallel_workflow(
        self, config: WorkflowConfiguration, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute workflow in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def execute_module_step(module_config):
            if module_config.module_type not in self.module_registry:
                raise ValueError(f"Module not registered: {module_config.module_type}")

            module = self.module_registry[module_config.module_type]
            module_inputs = self._prepare_module_inputs(module_config, context)

            try:
                start_time = time.time()
                result = module(**module_inputs)
                execution_time = time.time() - start_time

                return {
                    "module_id": module_config.module_id,
                    "module_type": module_config.module_type,
                    "result": result,
                    "execution_time": execution_time,
                    "success": True,
                }
            except Exception as e:
                return {
                    "module_id": module_config.module_id,
                    "module_type": module_config.module_type,
                    "error": str(e),
                    "success": False,
                }

        results = []
        with ThreadPoolExecutor(max_workers=len(config.modules)) as executor:
            future_to_config = {
                executor.submit(execute_module_step, module_config): module_config
                for module_config in config.modules
            }

            for future in as_completed(future_to_config):
                result = future.result()
                results.append(result)
                context.step_results.append(result)

        return {
            "execution_mode": "parallel",
            "steps": results,
            "final_variables": context.variables,
        }

    def _execute_conditional_workflow(
        self, config: WorkflowConfiguration, context: ExecutionContext
    ) -> dict[str, Any]:
        """Execute workflow with conditional logic"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated condition evaluation
        results = []

        for module_config in config.modules:
            # Check conditions
            should_execute = True
            if module_config.conditions:
                should_execute = self._evaluate_conditions(
                    module_config.conditions, context
                )

            if should_execute:
                context.current_step += 1

                if module_config.module_type not in self.module_registry:
                    raise ValueError(
                        f"Module not registered: {module_config.module_type}"
                    )

                module = self.module_registry[module_config.module_type]
                module_inputs = self._prepare_module_inputs(module_config, context)

                try:
                    start_time = time.time()
                    result = module(**module_inputs)
                    execution_time = time.time() - start_time

                    step_result = {
                        "step": context.current_step,
                        "module_id": module_config.module_id,
                        "module_type": module_config.module_type,
                        "result": result,
                        "execution_time": execution_time,
                        "success": True,
                        "executed": True,
                    }

                    self._update_context_variables(module_config, result, context)

                except Exception as e:
                    step_result = {
                        "step": context.current_step,
                        "module_id": module_config.module_id,
                        "module_type": module_config.module_type,
                        "error": str(e),
                        "success": False,
                        "executed": True,
                    }
            else:
                step_result = {
                    "module_id": module_config.module_id,
                    "module_type": module_config.module_type,
                    "executed": False,
                    "reason": "Condition not met",
                }

            results.append(step_result)
            context.step_results.append(step_result)

        return {
            "execution_mode": "conditional",
            "steps": results,
            "final_variables": context.variables,
        }

    def _prepare_module_inputs(
        self, module_config: ModuleConfiguration, context: ExecutionContext
    ) -> dict[str, Any]:
        """Prepare inputs for module execution"""
        inputs = {}

        # Apply input mappings
        for input_key, variable_key in module_config.input_mappings.items():
            if variable_key in context.variables:
                inputs[input_key] = context.variables[variable_key]

        # Add module parameters
        inputs.update(module_config.parameters)

        return inputs

    def _update_context_variables(
        self, module_config: ModuleConfiguration, result: Any, context: ExecutionContext
    ):
        """Update context variables with module results"""
        if isinstance(result, dict):
            # Apply output mappings
            for result_key, variable_key in module_config.output_mappings.items():
                if result_key in result:
                    context.variables[variable_key] = result[result_key]

            # Add all result keys if no specific mappings
            if not module_config.output_mappings:
                context.variables.update(result)
        else:
            # Store result with module ID as key
            context.variables[f"{module_config.module_id}_result"] = result

    def _evaluate_conditions(
        self, conditions: dict[str, Any], context: ExecutionContext
    ) -> bool:
        """Evaluate conditions for conditional execution"""
        # Simplified condition evaluation
        # In practice, you'd want a more sophisticated condition engine

        for condition_type, condition_value in conditions.items():
            if condition_type == "variable_exists":
                if condition_value not in context.variables:
                    return False
            elif condition_type == "variable_equals":
                var_name, expected_value = condition_value
                if context.variables.get(var_name) != expected_value:
                    return False
            elif condition_type == "step_count_greater_than":
                if context.current_step <= condition_value:
                    return False

        return True

    def get_execution_status(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Get status of a workflow execution"""
        if execution_id not in self.execution_contexts:
            return None

        context = self.execution_contexts[execution_id]
        return {
            "execution_id": execution_id,
            "workflow_id": context.workflow_id,
            "status": context.status.value,
            "current_step": context.current_step,
            "start_time": context.start_time,
            "elapsed_time": time.time() - context.start_time,
            "step_results_count": len(context.step_results),
            "error_count": len(context.error_log),
        }


class ConfigurationManager:
    """Manager for workflow and module configurations"""

    def __init__(self):
        self.configurations = {}

    def save_configuration(self, config: WorkflowConfiguration, filepath: str):
        """Save workflow configuration to file"""
        config_dict = asdict(config)

        # Convert enums to strings
        config_dict["execution_mode"] = config.execution_mode.value

        filepath = Path(filepath)

        if filepath.suffix.lower() == ".json":
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif filepath.suffix.lower() in [".yml", ".yaml"]:
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")

        logger.info("Configuration saved to %s", filepath)

    def load_configuration(self, filepath: str) -> WorkflowConfiguration:
        """Load workflow configuration from file"""
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".json":
            with open(filepath, "r") as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in [".yml", ".yaml"]:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")

        # Convert execution mode string back to enum
        if "execution_mode" in config_dict:
            config_dict["execution_mode"] = WorkflowExecutionMode(
                config_dict["execution_mode"]
            )

        # Convert module configurations
        if "modules" in config_dict:
            modules = []
            for module_dict in config_dict["modules"]:
                modules.append(ModuleConfiguration(**module_dict))
            config_dict["modules"] = modules

        config = WorkflowConfiguration(**config_dict)
        self.configurations[config.workflow_id] = config

        logger.info("Configuration loaded from %s", filepath)
        return config

    def export_module(self, module: CustomModuleBase, filepath: str):
        """Export a module to file for sharing"""
        filepath = Path(filepath)

        module_data = {
            "class_name": module.__class__.__name__,
            "module_name": module.__class__.__module__,
            "metadata": module.get_info() if hasattr(module, "get_info") else {},
            "serialized_module": pickle.dumps(
                module
            ).hex(),  # Hex encoding for JSON compatibility
        }

        with open(filepath, "w") as f:
            json.dump(module_data, f, indent=2, default=str)

        logger.info("Module exported to %s", filepath)

    def import_module(self, filepath: str) -> CustomModuleBase:
        """Import a module from file"""
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            module_data = json.load(f)

        # Deserialize the module
        serialized_data = bytes.fromhex(module_data["serialized_module"])
        module = pickle.loads(serialized_data)

        logger.info("Module imported to %s", filepath)
        return module


def demonstrate_module_composition():
    """Demonstrate the module composition and workflow system"""
    print("=== DSPy Module Composition and Workflow System Demonstration ===\n")

    # Import components for demonstration
    from component_library import (
        SentimentAnalyzerComponent,
        TextCleanerComponent,
        TextSummarizerComponent,
    )

    # Example 1: Module Composer
    print("1. Module Composer:")
    print("-" * 40)

    composer = ModuleComposer()

    # Register modules
    cleaner = TextCleanerComponent()
    sentiment = SentimentAnalyzerComponent()
    summarizer = TextSummarizerComponent()

    composer.register_module("cleaner", cleaner)
    composer.register_module("sentiment", sentiment)
    composer.register_module("summarizer", summarizer)

    # Create sequential composition
    sequential_comp = composer.create_sequential_composition(
        ["cleaner", "sentiment", "summarizer"], "Text Processing Pipeline"
    )

    test_text = "This is an amazing article about AI! <p>It contains HTML tags and multiple sentences.</p> The content is very informative and well-written."

    seq_result = sequential_comp(text=test_text)
    print(f"Sequential Composition:")
    print(f"  Steps: {seq_result['total_steps']}")
    print(f"  Successful: {seq_result['successful_steps']}")
    print(f"  Composition ID: {seq_result['composition_id']}")

    # Create parallel composition
    parallel_comp = composer.create_parallel_composition(
        ["sentiment", "summarizer"], "Parallel Analysis"
    )

    par_result = parallel_comp(text=test_text)
    print(f"\nParallel Composition:")
    print(f"  Steps: {par_result['total_steps']}")
    print(f"  Successful: {par_result['successful_steps']}")
    print(f"  Parallel execution time: {par_result['parallel_execution_time']:.3f}s")

    # Example 2: Conditional Composition
    print("\n2. Conditional Composition:")
    print("-" * 40)

    # Create condition function
    def is_long_text(**kwargs):
        return len(kwargs.get("text", "")) > 100

    conditional_comp = composer.create_conditional_composition(
        is_long_text, "summarizer", "sentiment", "Length-based Processing"
    )

    short_text = "Short message."
    long_text = "This is a very long text that contains many words and sentences. " * 5

    for text, label in [(short_text, "Short"), (long_text, "Long")]:
        cond_result = conditional_comp(text=text)
        print(f"{label} text:")
        print(f"  Condition result: {cond_result['condition_result']}")
        print(f"  Selected module: {cond_result['selected_module']}")
        print(f"  Success: {cond_result['success']}")

    # Example 3: Workflow Orchestrator
    print("\n3. Workflow Orchestrator:")
    print("-" * 40)

    orchestrator = WorkflowOrchestrator()

    # Register modules with orchestrator
    orchestrator.register_module("text_cleaner", cleaner)
    orchestrator.register_module("sentiment_analyzer", sentiment)
    orchestrator.register_module("text_summarizer", summarizer)

    # Create workflow configuration
    workflow_config = WorkflowConfiguration(
        workflow_id="text_processing_workflow",
        name="Complete Text Processing Workflow",
        description="Clean, analyze sentiment, and summarize text",
        execution_mode=WorkflowExecutionMode.SEQUENTIAL,
        modules=[
            ModuleConfiguration(
                module_id="clean_step",
                module_type="text_cleaner",
                input_mappings={"text": "input_text"},
                output_mappings={"cleaned_text": "cleaned_text"},
            ),
            ModuleConfiguration(
                module_id="sentiment_step",
                module_type="sentiment_analyzer",
                input_mappings={"text": "cleaned_text"},
                output_mappings={
                    "sentiment": "text_sentiment",
                    "confidence": "sentiment_confidence",
                },
            ),
            ModuleConfiguration(
                module_id="summary_step",
                module_type="text_summarizer",
                input_mappings={"text": "cleaned_text"},
                output_mappings={"summary": "text_summary"},
            ),
        ],
    )

    # Register and execute workflow
    orchestrator.register_workflow(workflow_config)

    workflow_inputs = {
        "input_text": "<p>This is a comprehensive article about artificial intelligence and machine learning technologies. The field has seen tremendous growth in recent years, with applications spanning from natural language processing to computer vision. Researchers and practitioners continue to push the boundaries of what's possible with AI systems.</p>"
    }

    workflow_result = orchestrator.execute_workflow(
        "text_processing_workflow", workflow_inputs
    )

    print(f"Workflow Execution:")
    print(f"  Execution ID: {workflow_result['execution_id']}")
    print(f"  Status: {workflow_result['status']}")
    print(
        f"  Steps completed: {workflow_result['performance_metrics']['steps_completed']}"
    )
    print(
        f"  Success rate: {workflow_result['performance_metrics']['success_rate']:.2%}"
    )
    print(
        f"  Total time: {workflow_result['performance_metrics']['total_execution_time']:.3f}s"
    )

    # Example 4: Configuration Management
    print("\n4. Configuration Management:")
    print("-" * 40)

    config_manager = ConfigurationManager()

    # Save configuration
    config_file = "08-custom-modules/sample_workflow_config.json"
    try:
        config_manager.save_configuration(workflow_config, config_file)
        print(f"✅ Configuration saved to {config_file}")

        # Load configuration
        loaded_config = config_manager.load_configuration(config_file)
        print(f"✅ Configuration loaded: {loaded_config.name}")
        print(f"   Modules: {len(loaded_config.modules)}")
        print(f"   Execution mode: {loaded_config.execution_mode.value}")

    except Exception as e:
        print(f"❌ Configuration management failed: {e}")

    # Example 5: Feedback Composition
    print("\n5. Feedback Composition:")
    print("-" * 40)

    def feedback_condition(variables, step_results):
        # Simple feedback condition: stop if sentiment is positive
        sentiment = variables.get("sentiment", "neutral")
        return sentiment == "positive"

    feedback_comp = composer.create_feedback_composition(
        ["cleaner", "sentiment"], feedback_condition, 3, "Feedback Processing"
    )

    feedback_result = feedback_comp(text="This is okay content that might improve.")
    print(f"Feedback Composition:")
    print(f"  Total iterations: {feedback_result['total_iterations']}")
    print(f"  Max iterations: {feedback_result['max_iterations']}")
    print(f"  Converged: {feedback_result['converged']}")


if __name__ == "__main__":
    """
    DSPy Module Composition and Workflow System Demonstration

    This script demonstrates:
    1. Module composition patterns (sequential, parallel, conditional, feedback)
    2. Workflow orchestration and execution management
    3. Configuration management and serialization
    4. Complex workflow execution with error handling
    5. Module export/import capabilities
    """

    try:
        demonstrate_module_composition()

        print(
            "\n✅ Module composition and workflow system demonstration completed successfully!"
        )
        print("\nKey takeaways:")
        print(
            "- Module composition enables building complex systems from simple components"
        )
        print("- Workflow orchestration provides structured execution management")
        print("- Configuration management enables workflow reusability and sharing")
        print("- Different execution modes support various processing patterns")
        print("- Feedback compositions enable iterative and adaptive processing")
        print("- Serialization capabilities support module and workflow sharing")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Module composition demonstration failed")
