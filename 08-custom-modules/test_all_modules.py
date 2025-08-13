#!/usr/bin/env python3
"""
Quick test script to verify all Module 08 components are working
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure DSPy with available API keys
from dspy_config import configure_dspy_lm, get_configured_model_info

dspy_configured = configure_dspy_lm()
if dspy_configured:
    model_info = get_configured_model_info()
    print(f"✅ DSPy configured with {model_info['provider']} {model_info['model']}")
else:
    print("ℹ️  Using fallback implementations (no API keys configured)")


def test_custom_module_template():
    """Test the custom module template"""
    print("Testing custom_module_template.py...")
    try:
        from custom_module_template import (
            CustomModuleBase,
            ModuleMetadata,
            ModuleValidator,
        )

        # Create a simple test module
        class TestModule(CustomModuleBase):
            def __init__(self):
                metadata = ModuleMetadata(
                    name="Test Module",
                    version="1.0.0",
                    description="Test module",
                    author="Test",
                )
                super().__init__(metadata)

            def forward(self, **kwargs):
                return {"result": "test"}

        module = TestModule()
        result = module(test="input")

        print(f"✅ custom_module_template.py - Result: {result}")
        return True
    except Exception as e:
        print(f"❌ custom_module_template.py - Error: {e}")
        return False


def test_component_library():
    """Test the component library"""
    print("Testing component_library.py...")
    try:
        from component_library import SentimentAnalyzerComponent, TextCleanerComponent

        cleaner = TextCleanerComponent()
        result = cleaner(text="<p>Hello, world!</p>")

        print(f"✅ component_library.py - Cleaned text: {result['cleaned_text']}")
        return True
    except Exception as e:
        print(f"❌ component_library.py - Error: {e}")
        return False


def test_module_testing_framework():
    """Test the module testing framework"""
    print("Testing module_testing_framework.py...")
    try:
        from component_library import TextCleanerComponent
        from module_testing_framework import ModuleTestRunner, TestCase, TestSuite

        module = TextCleanerComponent()
        test_case = TestCase(
            name="simple_test", inputs={"text": "test"}, description="Simple test"
        )

        test_runner = ModuleTestRunner(module)
        result = test_runner.run_test_case(test_case)

        print(f"✅ module_testing_framework.py - Test success: {result.success}")
        return True
    except Exception as e:
        print(f"❌ module_testing_framework.py - Error: {e}")
        return False


def test_module_composition():
    """Test the module composition"""
    print("Testing module_composition.py...")
    try:
        from component_library import SentimentAnalyzerComponent, TextCleanerComponent
        from module_composition import SequentialComposition

        modules = [TextCleanerComponent(), SentimentAnalyzerComponent()]
        composition = SequentialComposition(modules, "Test Composition")

        result = composition(text="Test text")

        print(f"✅ module_composition.py - Composition steps: {result['total_steps']}")
        return True
    except Exception as e:
        print(f"❌ module_composition.py - Error: {e}")
        return False


if __name__ == "__main__":
    print("=== Module 08: Quick Component Test ===\n")

    tests = [
        test_custom_module_template,
        test_component_library,
        test_module_testing_framework,
        test_module_composition,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Module 08 components are working correctly!")
    else:
        print("⚠️  Some components have issues that need to be addressed.")
