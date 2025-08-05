"""
DSPy extension utilities and wrappers for enhanced functionality.

This module provides reactive DSPy module integration for Marimo,
signature testing utilities, and optimization tracking components.
"""

# Standard Library
import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# Third-Party Library
import dspy

# Local Modules
from .config import get_config
from .utils import ProgressTracker

logger = logging.getLogger(__name__)


# =============================================================================
# Reactive DSPy Module Wrappers
# =============================================================================


class ReactiveModule:
    """
    Wrapper for DSPy modules that provides reactive updates for Marimo integration.

    This class wraps any DSPy module and provides hooks for parameter changes,
    execution tracking, and result caching.
    """

    def __init__(
        self,
        module: dspy.Module,
        name: Optional[str] = None,
        enable_caching: bool = True,
        enable_tracing: bool = True,
    ):
        """
        Initialize the reactive module wrapper.

        Args:
            module: DSPy module to wrap
            name: Optional name for the module
            enable_caching: Enable result caching
            enable_tracing: Enable execution tracing
        """
        self.module = module
        self.name = name or module.__class__.__name__
        self.enable_caching = enable_caching
        self.enable_tracing = enable_tracing

        # Execution tracking
        self.execution_history = []
        self.parameter_history = []
        self.cache = {}

        # Callbacks
        self.on_parameter_change = []
        self.on_execution_start = []
        self.on_execution_complete = []
        self.on_error = []

        # Thread safety
        self._lock = threading.Lock()

    def add_callback(self, event: str, callback: Callable):
        """
        Add a callback for specific events.

        Args:
            event: Event type ('parameter_change', 'execution_start', 'execution_complete', 'error')
            callback: Callback function
        """
        if event == "parameter_change":
            self.on_parameter_change.append(callback)
        elif event == "execution_start":
            self.on_execution_start.append(callback)
        elif event == "execution_complete":
            self.on_execution_complete.append(callback)
        elif event == "error":
            self.on_error.append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")

    def update_parameters(self, **kwargs):
        """
        Update module parameters and trigger callbacks.

        Args:
            **kwargs: Parameter updates
        """
        with self._lock:
            # Store parameter change
            param_change = {"timestamp": datetime.now(), "parameters": kwargs.copy()}
            self.parameter_history.append(param_change)

            # Apply parameters to module if possible
            for key, value in kwargs.items():
                if hasattr(self.module, key):
                    setattr(self.module, key, value)

            # Trigger callbacks
            for callback in self.on_parameter_change:
                try:
                    callback(self, kwargs)
                except Exception as e:
                    logger.error(f"Parameter change callback error: {e}")

    def __call__(self, *args, **kwargs):
        """Execute the wrapped module with tracking."""
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        """
        Execute the module with full tracking and caching.

        Args:
            *args: Positional arguments for the module
            **kwargs: Keyword arguments for the module

        Returns:
            Module execution result
        """
        # Create cache key
        cache_key = None
        if self.enable_caching:
            cache_key = self._create_cache_key(args, kwargs)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {self.name}")
                return self.cache[cache_key]["result"]

        # Trigger execution start callbacks
        execution_id = len(self.execution_history)
        start_time = time.time()

        for callback in self.on_execution_start:
            try:
                callback(self, execution_id, args, kwargs)
            except Exception as e:
                logger.error(f"Execution start callback error: {e}")

        try:
            # Execute the module
            if self.enable_tracing:
                with dspy.context(trace=True):
                    result = self.module(*args, **kwargs)
            else:
                result = self.module(*args, **kwargs)

            end_time = time.time()
            execution_time = end_time - start_time

            # Store execution record
            execution_record = {
                "id": execution_id,
                "timestamp": datetime.now(),
                "args": args,
                "kwargs": kwargs,
                "result": result,
                "execution_time": execution_time,
                "success": True,
            }

            with self._lock:
                self.execution_history.append(execution_record)

                # Cache result
                if self.enable_caching and cache_key:
                    self.cache[cache_key] = {
                        "result": result,
                        "timestamp": datetime.now(),
                        "execution_time": execution_time,
                    }

            # Trigger completion callbacks
            for callback in self.on_execution_complete:
                try:
                    callback(self, execution_id, result, execution_time)
                except Exception as e:
                    logger.error(f"Execution complete callback error: {e}")

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            # Store error record
            error_record = {
                "id": execution_id,
                "timestamp": datetime.now(),
                "args": args,
                "kwargs": kwargs,
                "error": str(e),
                "execution_time": execution_time,
                "success": False,
            }

            with self._lock:
                self.execution_history.append(error_record)

            # Trigger error callbacks
            for callback in self.on_error:
                try:
                    callback(self, execution_id, e, execution_time)
                except Exception as err:
                    logger.error(f"Error callback error: {err}")

            raise

    def _create_cache_key(self, args: Tuple, kwargs: Dict) -> str:
        """Create a cache key from arguments."""
        try:
            key_data = {
                "args": args,
                "kwargs": kwargs,
                "module_class": self.module.__class__.__name__,
            }
            return json.dumps(key_data, sort_keys=True, default=str)
        except Exception:
            # Fallback to string representation
            return f"{self.module.__class__.__name__}_{hash((args, tuple(kwargs.items())))}"

    def clear_cache(self):
        """Clear the execution cache."""
        with self._lock:
            self.cache.clear()

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            if not self.execution_history:
                return {}

            successful_executions = [e for e in self.execution_history if e["success"]]
            failed_executions = [e for e in self.execution_history if not e["success"]]

            execution_times = [e["execution_time"] for e in successful_executions]

            stats = {
                "total_executions": len(self.execution_history),
                "successful_executions": len(successful_executions),
                "failed_executions": len(failed_executions),
                "success_rate": (
                    len(successful_executions) / len(self.execution_history)
                    if self.execution_history
                    else 0
                ),
                "cache_hits": len(self.cache),
                "avg_execution_time": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
            }

            return stats


def make_reactive(module: dspy.Module, **kwargs) -> ReactiveModule:
    """
    Convert a DSPy module to a reactive module.

    Args:
        module: DSPy module to make reactive
        **kwargs: Additional arguments for ReactiveModule

    Returns:
        ReactiveModule wrapper
    """
    return ReactiveModule(module, **kwargs)


# =============================================================================
# Signature Testing Utilities
# =============================================================================


class SignatureTester:
    """
    Utility for testing DSPy signatures with various inputs and configurations.
    """

    def __init__(self, signature: Type[dspy.Signature]):
        """
        Initialize the signature tester.

        Args:
            signature: DSPy signature class to test
        """
        self.signature = signature
        self.test_results = []
        self.predictor = dspy.Predict(signature)

    def test_single(self, **inputs) -> Dict[str, Any]:
        """
        Test the signature with a single set of inputs.

        Args:
            **inputs: Input values for the signature

        Returns:
            Test result dictionary
        """
        start_time = time.time()

        try:
            result = self.predictor(**inputs)
            end_time = time.time()

            test_result = {
                "timestamp": datetime.now(),
                "inputs": inputs,
                "outputs": {
                    key: getattr(result, key)
                    for key in result.__dict__
                    if not key.startswith("_")
                },
                "execution_time": end_time - start_time,
                "success": True,
                "error": None,
            }

        except Exception as e:
            end_time = time.time()
            test_result = {
                "timestamp": datetime.now(),
                "inputs": inputs,
                "outputs": {},
                "execution_time": end_time - start_time,
                "success": False,
                "error": str(e),
            }

        self.test_results.append(test_result)
        return test_result

    def test_batch(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Test the signature with multiple input sets.

        Args:
            input_list: List of input dictionaries

        Returns:
            List of test results
        """
        results = []
        progress = ProgressTracker(len(input_list), "Testing signature")

        for inputs in input_list:
            result = self.test_single(**inputs)
            results.append(result)
            progress.update()

        return results

    def test_variations(
        self, base_inputs: Dict[str, Any], variations: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Test signature with variations of base inputs.

        Args:
            base_inputs: Base input values
            variations: Dictionary mapping input names to lists of alternative values

        Returns:
            List of test results
        """
        from itertools import product

        # Generate all combinations
        variation_keys = list(variations.keys())
        variation_values = list(variations.values())

        test_inputs = []
        for combo in product(*variation_values):
            inputs = base_inputs.copy()
            for key, value in zip(variation_keys, combo):
                inputs[key] = value
            test_inputs.append(inputs)

        return self.test_batch(test_inputs)

    def get_success_rate(self) -> float:
        """Get the success rate of all tests."""
        if not self.test_results:
            return 0.0

        successful = sum(1 for result in self.test_results if result["success"])
        return successful / len(self.test_results)

    def get_average_execution_time(self) -> float:
        """Get the average execution time of successful tests."""
        successful_results = [r for r in self.test_results if r["success"]]
        if not successful_results:
            return 0.0

        total_time = sum(r["execution_time"] for r in successful_results)
        return total_time / len(successful_results)

    def clear_results(self):
        """Clear all test results."""
        self.test_results.clear()


# =============================================================================
# Optimization Tracking
# =============================================================================


class OptimizationTracker:
    """
    Utility for tracking DSPy optimization progress and results.
    """

    def __init__(self, optimizer_name: str = "Unknown"):
        """
        Initialize the optimization tracker.

        Args:
            optimizer_name: Name of the optimizer being tracked
        """
        self.optimizer_name = optimizer_name
        self.optimization_runs = []
        self.current_run = None
        self.callbacks = []

    def start_run(
        self, config: Dict[str, Any], trainset_size: int, metric_name: str = "score"
    ) -> str:
        """
        Start tracking a new optimization run.

        Args:
            config: Optimization configuration
            trainset_size: Size of training set
            metric_name: Name of the metric being optimized

        Returns:
            Run ID
        """
        run_id = f"run_{len(self.optimization_runs)}"

        self.current_run = {
            "id": run_id,
            "optimizer": self.optimizer_name,
            "config": config,
            "trainset_size": trainset_size,
            "metric_name": metric_name,
            "start_time": datetime.now(),
            "end_time": None,
            "steps": [],
            "final_score": None,
            "success": False,
        }

        return run_id

    def log_step(
        self,
        step: int,
        score: float,
        additional_metrics: Optional[Dict[str, float]] = None,
        module_state: Optional[Any] = None,
    ):
        """
        Log an optimization step.

        Args:
            step: Step number
            score: Primary metric score
            additional_metrics: Additional metrics
            module_state: Current module state (optional)
        """
        if not self.current_run:
            logger.warning("No active optimization run")
            return

        step_data = {
            "step": step,
            "timestamp": datetime.now(),
            "score": score,
            "additional_metrics": additional_metrics or {},
            "module_state": module_state,
        }

        self.current_run["steps"].append(step_data)

        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback("step", self.current_run["id"], step_data)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")

    def end_run(self, final_score: float, success: bool = True):
        """
        End the current optimization run.

        Args:
            final_score: Final optimization score
            success: Whether optimization was successful
        """
        if not self.current_run:
            logger.warning("No active optimization run")
            return

        self.current_run["end_time"] = datetime.now()
        self.current_run["final_score"] = final_score
        self.current_run["success"] = success

        # Calculate duration
        duration = self.current_run["end_time"] - self.current_run["start_time"]
        self.current_run["duration"] = duration.total_seconds()

        self.optimization_runs.append(self.current_run)

        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback("run_complete", self.current_run["id"], self.current_run)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")

        self.current_run = None

    def add_callback(self, callback: Callable):
        """
        Add a callback for optimization events.

        Args:
            callback: Callback function that receives (event_type, run_id, data)
        """
        self.callbacks.append(callback)

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        """Get the optimization run with the best final score."""
        if not self.optimization_runs:
            return None

        successful_runs = [run for run in self.optimization_runs if run["success"]]
        if not successful_runs:
            return None

        return max(successful_runs, key=lambda x: x["final_score"])

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization runs."""
        if not self.optimization_runs:
            return {}

        successful_runs = [run for run in self.optimization_runs if run["success"]]

        summary = {
            "total_runs": len(self.optimization_runs),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(self.optimization_runs),
            "optimizer": self.optimizer_name,
        }

        if successful_runs:
            scores = [run["final_score"] for run in successful_runs]
            durations = [run["duration"] for run in successful_runs]

            summary.update(
                {
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "avg_score": sum(scores) / len(scores),
                    "avg_duration": sum(durations) / len(durations),
                }
            )

        return summary


# =============================================================================
# DSPy Configuration Helpers
# =============================================================================


def configure_dspy_lm(
    provider: Optional[str] = None, model: Optional[str] = None, **kwargs
) -> dspy.LM:
    """
    Configure DSPy language model from application config.

    Args:
        provider: LLM provider ('openai', 'anthropic', 'cohere')
        model: Model name
        **kwargs: Additional model parameters

    Returns:
        Configured DSPy LM instance
    """
    config = get_config()

    if provider is None:
        provider = config.default_llm_provider

    if model is None:
        model = config.default_model

    # Get provider-specific configuration
    llm_config = config.get_llm_config(provider)
    llm_config.update(kwargs)

    # Create LM instance
    if provider == "openai":
        lm = dspy.LM(f"openai/{model}", **llm_config)
    elif provider == "anthropic":
        lm = dspy.LM(f"anthropic/{model}", **llm_config)
    elif provider == "cohere":
        lm = dspy.LM(f"cohere/{model}", **llm_config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return lm


def setup_dspy_environment(
    provider: Optional[str] = None, model: Optional[str] = None, **kwargs
):
    """
    Set up DSPy environment with configured language model.

    Args:
        provider: LLM provider
        model: Model name
        **kwargs: Additional configuration
    """
    lm = configure_dspy_lm(provider, model, **kwargs)
    dspy.configure(lm=lm)

    logger.info(f"DSPy configured with {provider}/{model}")


@contextmanager
def dspy_context(**kwargs):
    """
    Context manager for temporary DSPy configuration.

    Args:
        **kwargs: DSPy configuration parameters
    """
    # Save current configuration
    old_config = dspy.settings

    try:
        # Apply temporary configuration
        dspy.configure(**kwargs)
        yield
    finally:
        # Restore original configuration
        dspy.settings = old_config


# =============================================================================
# Module Composition Utilities
# =============================================================================


class ModuleChain:
    """
    Utility for chaining multiple DSPy modules together.
    """

    def __init__(self, modules: List[dspy.Module], names: Optional[List[str]] = None):
        """
        Initialize the module chain.

        Args:
            modules: List of DSPy modules to chain
            names: Optional names for the modules
        """
        self.modules = modules
        self.names = names or [f"module_{i}" for i in range(len(modules))]
        self.execution_history = []

    def __call__(self, *args, **kwargs):
        """Execute the module chain."""
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        """
        Execute all modules in the chain.

        Args:
            *args: Initial arguments
            **kwargs: Initial keyword arguments

        Returns:
            Final result from the chain
        """
        start_time = time.time()
        execution_record = {
            "timestamp": datetime.now(),
            "initial_args": args,
            "initial_kwargs": kwargs,
            "steps": [],
        }

        current_args = args
        current_kwargs = kwargs

        try:
            for i, (module, name) in enumerate(zip(self.modules, self.names)):
                step_start = time.time()

                # Execute module
                result = module(*current_args, **current_kwargs)

                step_end = time.time()

                # Record step
                step_record = {
                    "module_index": i,
                    "module_name": name,
                    "input_args": current_args,
                    "input_kwargs": current_kwargs,
                    "result": result,
                    "execution_time": step_end - step_start,
                }
                execution_record["steps"].append(step_record)

                # Prepare inputs for next module
                if hasattr(result, "__dict__"):
                    # Use result attributes as kwargs for next module
                    current_args = ()
                    current_kwargs = {
                        k: v
                        for k, v in result.__dict__.items()
                        if not k.startswith("_")
                    }
                else:
                    # Use result as first argument
                    current_args = (result,)
                    current_kwargs = {}

            end_time = time.time()
            execution_record["total_time"] = end_time - start_time
            execution_record["success"] = True
            execution_record["final_result"] = result

            self.execution_history.append(execution_record)
            return result

        except Exception as e:
            end_time = time.time()
            execution_record["total_time"] = end_time - start_time
            execution_record["success"] = False
            execution_record["error"] = str(e)

            self.execution_history.append(execution_record)
            raise

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for the chain."""
        if not self.execution_history:
            return {}

        successful_executions = [e for e in self.execution_history if e["success"]]

        stats = {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "avg_total_time": (
                sum(e["total_time"] for e in successful_executions)
                / len(successful_executions)
                if successful_executions
                else 0
            ),
        }

        # Per-module stats
        module_stats = {}
        for i, name in enumerate(self.names):
            module_times = []
            for execution in successful_executions:
                if i < len(execution["steps"]):
                    module_times.append(execution["steps"][i]["execution_time"])

            if module_times:
                module_stats[name] = {
                    "avg_time": sum(module_times) / len(module_times),
                    "min_time": min(module_times),
                    "max_time": max(module_times),
                }

        stats["module_stats"] = module_stats
        return stats
