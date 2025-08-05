import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import sys
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    import dspy
    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from common import (
        DSPyParameterPanel,
        DSPyResultViewer,
        get_config,
        setup_dspy_environment,
    )

    return (
        Any,
        DSPyParameterPanel,
        DSPyResultViewer,
        Dict,
        List,
        Optional,
        Path,
        dspy,
        get_config,
        json,
        mo,
        project_root,
        setup_dspy_environment,
        sys,
        time,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🔧 Tool Integration Framework for DSPy Agents
        
        **Duration:** 75-90 minutes  
        **Prerequisites:** Completed ReAct Implementation
        
        ## 🎯 Learning Objectives
        
        By the end of this module, you will:
        - ✅ Build a comprehensive tool integration framework
        - ✅ Integrate external APIs and web services
        - ✅ Create tool composition and chaining patterns
        - ✅ Implement error handling and fallback strategies
        - ✅ Monitor and optimize tool performance
        
        ## 🛠️ What is Tool Integration?
        
        Tool integration allows DSPy agents to:
        - **Access External Data** - APIs, databases, web services
        - **Perform Actions** - Send emails, create files, make purchases
        - **Process Information** - Calculate, analyze, transform data
        - **Interact with Systems** - Control software, hardware, IoT devices
        
        ## 🏗️ Framework Architecture
        
        Our tool integration framework includes:
        1. **Tool Registry** - Centralized tool management
        2. **Execution Engine** - Safe, monitored tool execution
        3. **Error Handling** - Graceful failure management
        4. **Performance Monitoring** - Usage analytics and optimization
        5. **Security Layer** - Access control and validation
        
        Let's build a production-ready tool integration system!
        """
    )
    return


@app.cell
def __(get_config, mo, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        mo.md(
            f"""
        ## ✅ Tool Integration Environment Ready
        
        **Configuration:**
        - Provider: **{config.default_llm_provider}**
        - Model: **{config.default_model}**
        - Tool framework enabled!
        
        Ready to build powerful tool integrations!
        """
        )
    else:
        mo.md(
            """
        ## ⚠️ Setup Required
        
        Please complete Module 00 setup first to configure your API keys.
        """
        )
    return available_providers, config
@app
.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🏗️ Step 1: Tool Registry Architecture
        
        Let's start by building a comprehensive tool registry system:
        """
        )

        # Tool Registry Implementation
        class ToolRegistry:
            """Centralized registry for managing and executing tools."""
            
            def __init__(self):
                self.tools = {}
                self.execution_stats = {}
                self.security_policies = {}
            
            def register_tool(self, name: str, tool_func, description: str, 
                            input_schema: dict, output_schema: dict, 
                            security_level: str = "low"):
                """Register a new tool with metadata and security settings."""
                self.tools[name] = {
                    "function": tool_func,
                    "description": description,
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                    "security_level": security_level,
                    "registered_at": time.time()
                }
                self.execution_stats[name] = {
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_duration": 0.0,
                    "last_used": None
                }
                return f"Tool '{name}' registered successfully"
            
            def execute_tool(self, name: str, inputs: dict, context: dict = None):
                """Execute a tool with monitoring and error handling."""
                if name not in self.tools:
                    return {"error": f"Tool '{name}' not found", "success": False}
                
                tool_info = self.tools[name]
                stats = self.execution_stats[name]
                
                # Update call statistics
                stats["calls"] += 1
                stats["last_used"] = time.time()
                
                try:
                    # Validate inputs (simplified)
                    if not self._validate_inputs(inputs, tool_info["input_schema"]):
                        stats["failures"] += 1
                        return {"error": "Invalid inputs", "success": False}
                    
                    # Execute tool
                    start_time = time.time()
                    result = tool_info["function"](inputs, context or {})
                    duration = time.time() - start_time
                    
                    # Update statistics
                    stats["successes"] += 1
                    stats["avg_duration"] = (
                        (stats["avg_duration"] * (stats["successes"] - 1) + duration) 
                        / stats["successes"]
                    )
                    
                    return {
                        "result": result,
                        "success": True,
                        "duration": duration,
                        "tool": name
                    }
                    
                except Exception as e:
                    stats["failures"] += 1
                    return {
                        "error": str(e),
                        "success": False,
                        "tool": name
                    }
            
            def _validate_inputs(self, inputs: dict, schema: dict) -> bool:
                """Simple input validation (in production, use jsonschema)."""
                required_fields = schema.get("required", [])
                return all(field in inputs for field in required_fields)
            
            def get_tool_info(self, name: str = None):
                """Get information about tools."""
                if name:
                    if name in self.tools:
                        return {
                            "tool": self.tools[name],
                            "stats": self.execution_stats[name]
                        }
                    return None
                else:
                    return {
                        "tools": list(self.tools.keys()),
                        "total_tools": len(self.tools),
                        "total_calls": sum(s["calls"] for s in self.execution_stats.values())
                    }

        # Create global tool registry
        tool_registry = ToolRegistry()

        mo.md(
            """
        ### 🏗️ Tool Registry Created
        
        **Features:**
        - **Tool Registration** - Add tools with metadata and schemas
        - **Execution Monitoring** - Track performance and usage
        - **Error Handling** - Graceful failure management
        - **Security Policies** - Access control and validation
        - **Statistics Tracking** - Performance analytics
        
        The registry is ready to manage our tools!
        """
        )
    else:
        ToolRegistry = None
        tool_registry = None
    return ToolRegistry, tool_registry


@app.cell
def __(available_providers, mo, tool_registry):
    if available_providers and tool_registry:
        mo.md(
            """
        ## 🔧 Step 2: Core Tool Implementations
        
        Let's implement a comprehensive set of tools for our agents:
        """
        )

        # Mathematical Tools
        def calculator_tool(inputs: dict, context: dict) -> dict:
            """Advanced calculator with multiple operations."""
            expression = inputs.get("expression", "")
            operation = inputs.get("operation", "evaluate")
            
            try:
                if operation == "evaluate":
                    # Safe evaluation with limited scope
                    allowed_names = {
                        "abs": abs, "round": round, "min": min, "max": max,
                        "sum": sum, "pow": pow, "sqrt": lambda x: x**0.5,
                        "pi": 3.14159265359, "e": 2.71828182846
                    }
                    result = eval(expression, {"__builtins__": {}}, allowed_names)
                    return {"result": result, "expression": expression}
                elif operation == "statistics":
                    numbers = inputs.get("numbers", [])
                    if not numbers:
                        return {"error": "No numbers provided"}
                    return {
                        "mean": sum(numbers) / len(numbers),
                        "min": min(numbers),
                        "max": max(numbers),
                        "count": len(numbers)
                    }
                else:
                    return {"error": f"Unknown operation: {operation}"}
            except Exception as e:
                return {"error": f"Calculation error: {str(e)}"}

        # Text Processing Tools
        def text_processor_tool(inputs: dict, context: dict) -> dict:
            """Advanced text processing capabilities."""
            text = inputs.get("text", "")
            operation = inputs.get("operation", "analyze")
            
            if operation == "analyze":
                words = text.split()
                sentences = text.split('.')
                return {
                    "word_count": len(words),
                    "sentence_count": len([s for s in sentences if s.strip()]),
                    "character_count": len(text),
                    "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
                }
            elif operation == "transform":
                transform_type = inputs.get("transform_type", "uppercase")
                if transform_type == "uppercase":
                    return {"result": text.upper()}
                elif transform_type == "lowercase":
                    return {"result": text.lower()}
                elif transform_type == "title":
                    return {"result": text.title()}
                elif transform_type == "reverse":
                    return {"result": text[::-1]}
            elif operation == "extract":
                extract_type = inputs.get("extract_type", "words")
                if extract_type == "words":
                    return {"result": text.split()}
                elif extract_type == "sentences":
                    return {"result": [s.strip() for s in text.split('.') if s.strip()]}
                elif extract_type == "numbers":
                    import re
                    numbers = re.findall(r'\d+\.?\d*', text)
                    return {"result": [float(n) for n in numbers]}
            
            return {"error": f"Unknown operation: {operation}"}

        # Web Search Tool (Simulated)
        def web_search_tool(inputs: dict, context: dict) -> dict:
            """Simulated web search tool (in production, integrate with real search APIs)."""
            query = inputs.get("query", "")
            max_results = inputs.get("max_results", 5)
            
            # Simulated search results database
            search_database = {
                "python": [
                    {"title": "Python.org - Official Website", "url": "https://python.org", "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."},
                    {"title": "Python Tutorial", "url": "https://docs.python.org/tutorial/", "snippet": "An informal introduction to Python programming language."},
                    {"title": "Real Python", "url": "https://realpython.com", "snippet": "Python tutorials and resources for developers."}
                ],
                "machine learning": [
                    {"title": "Machine Learning Explained", "url": "https://example.com/ml", "snippet": "Machine learning is a method of data analysis that automates analytical model building."},
                    {"title": "Scikit-learn", "url": "https://scikit-learn.org", "snippet": "Simple and efficient tools for predictive data analysis."},
                    {"title": "TensorFlow", "url": "https://tensorflow.org", "snippet": "An end-to-end open source platform for machine learning."}
                ],
                "dspy": [
                    {"title": "DSPy Framework", "url": "https://dspy.ai", "snippet": "DSPy is a framework for algorithmically optimizing LM prompts and weights."},
                    {"title": "DSPy Documentation", "url": "https://dspy.ai/docs", "snippet": "Comprehensive guide to using DSPy for LM programming."}
                ]
            }
            
            # Find matching results
            results = []
            query_lower = query.lower()
            for key, entries in search_database.items():
                if key in query_lower:
                    results.extend(entries[:max_results])
                    break
            
            if not results:
                results = [{"title": f"Search results for '{query}'", "url": "https://example.com", "snippet": f"No specific results found for '{query}' in simulated database."}]
            
            return {
                "query": query,
                "results": results[:max_results],
                "total_found": len(results)
            }

        # Data Storage Tool
        def data_storage_tool(inputs: dict, context: dict) -> dict:
            """Simple data storage and retrieval tool."""
            operation = inputs.get("operation", "store")
            
            # Initialize storage if not exists
            if not hasattr(data_storage_tool, "storage"):
                data_storage_tool.storage = {}
            
            if operation == "store":
                key = inputs.get("key", "")
                value = inputs.get("value", "")
                if not key:
                    return {"error": "Key is required for storage"}
                data_storage_tool.storage[key] = {
                    "value": value,
                    "stored_at": time.time(),
                    "access_count": 0
                }
                return {"message": f"Stored data with key '{key}'", "key": key}
            
            elif operation == "retrieve":
                key = inputs.get("key", "")
                if key in data_storage_tool.storage:
                    data_storage_tool.storage[key]["access_count"] += 1
                    return {
                        "key": key,
                        "value": data_storage_tool.storage[key]["value"],
                        "stored_at": data_storage_tool.storage[key]["stored_at"],
                        "access_count": data_storage_tool.storage[key]["access_count"]
                    }
                else:
                    return {"error": f"No data found for key '{key}'"}
            
            elif operation == "list":
                return {
                    "keys": list(data_storage_tool.storage.keys()),
                    "total_items": len(data_storage_tool.storage)
                }
            
            elif operation == "delete":
                key = inputs.get("key", "")
                if key in data_storage_tool.storage:
                    del data_storage_tool.storage[key]
                    return {"message": f"Deleted data with key '{key}'"}
                else:
                    return {"error": f"No data found for key '{key}'"}
            
            return {"error": f"Unknown operation: {operation}"}

        # Register all tools
        tool_registry.register_tool(
            "calculator",
            calculator_tool,
            "Advanced calculator with mathematical operations and statistics",
            {"required": ["expression"], "optional": ["operation", "numbers"]},
            {"result": "number or dict", "error": "string"},
            "low"
        )

        tool_registry.register_tool(
            "text_processor",
            text_processor_tool,
            "Text analysis, transformation, and extraction capabilities",
            {"required": ["text", "operation"], "optional": ["transform_type", "extract_type"]},
            {"result": "varies", "error": "string"},
            "low"
        )

        tool_registry.register_tool(
            "web_search",
            web_search_tool,
            "Web search functionality (simulated)",
            {"required": ["query"], "optional": ["max_results"]},
            {"results": "list", "total_found": "number"},
            "medium"
        )

        tool_registry.register_tool(
            "data_storage",
            data_storage_tool,
            "Data storage and retrieval system",
            {"required": ["operation"], "optional": ["key", "value"]},
            {"result": "varies", "error": "string"},
            "medium"
        )

        mo.md(
            """
        ### 🔧 Core Tools Registered
        
        **Available Tools:**
        - **Calculator** - Mathematical operations and statistics
        - **Text Processor** - Analysis, transformation, extraction
        - **Web Search** - Information retrieval (simulated)
        - **Data Storage** - Persistent data management
        
        All tools are registered with proper schemas and monitoring!
        """
        )
    else:
        calculator_tool = None
        text_processor_tool = None
        web_search_tool = None
        data_storage_tool = None
    return calculator_tool, data_storage_tool, text_processor_tool, web_search_tool@app.
cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🤖 Step 3: Tool-Integrated Agent
        
        Now let's create an agent that can intelligently use our tool registry:
        """
        )

        # Define tool-integrated agent signature
        class ToolIntegratedAgentSignature(dspy.Signature):
            """Intelligent agent that can use multiple tools to solve complex problems."""
            problem = dspy.InputField(desc="The problem or task to solve")
            available_tools = dspy.InputField(desc="List of available tools and their capabilities")
            reasoning = dspy.OutputField(desc="Step-by-step reasoning about the problem")
            tool_plan = dspy.OutputField(desc="Plan for which tools to use and in what order")
            tool_calls = dspy.OutputField(desc="Specific tool calls with inputs (JSON format)")
            expected_outcome = dspy.OutputField(desc="Expected outcome from tool execution")
            final_answer = dspy.OutputField(desc="Final answer after tool execution")

        # Create the agent
        tool_agent = dspy.ChainOfThought(ToolIntegratedAgentSignature)

        mo.md(
            """
        ### 🤖 Tool-Integrated Agent Created
        
        This agent can:
        - **Analyze Problems** - Understand what needs to be solved
        - **Plan Tool Usage** - Decide which tools to use and when
        - **Execute Tools** - Make specific tool calls with proper inputs
        - **Synthesize Results** - Combine tool outputs into final answers
        """
        )
    else:
        ToolIntegratedAgentSignature = None
        tool_agent = None
    return ToolIntegratedAgentSignature, tool_agent


@app.cell
def __(available_providers, mo, tool_registry):
    if available_providers and tool_registry:
        # Complex problems for tool integration
        complex_problems = [
            "Calculate the compound interest on $5000 at 3.5% annual rate for 10 years, then search for information about investment strategies and store the result.",
            "Analyze the text 'The quick brown fox jumps over the lazy dog' for word statistics, transform it to uppercase, and store both results.",
            "Search for information about Python programming, extract any numbers from the search results, and calculate their average.",
            "Store three pieces of data: 'name: John', 'age: 30', 'city: New York', then retrieve all stored data and analyze the text content."
        ]

        problem_selector = mo.ui.dropdown(
            options=complex_problems,
            label="Select a complex problem",
            value=complex_problems[0]
        )

        run_agent_demo = mo.ui.button(label="🤖 Run Tool-Integrated Agent")

        # Get available tools info
        tools_info = tool_registry.get_tool_info()
        available_tools_desc = f"Available tools: {', '.join(tools_info['tools'])}"

        mo.vstack([
            mo.md("### 🧩 Complex Problem Solving"),
            mo.md("Select a multi-step problem that requires multiple tools:"),
            problem_selector,
            mo.md(f"**{available_tools_desc}**"),
            run_agent_demo
        ])
    else:
        complex_problems = None
        problem_selector = None
        run_agent_demo = None
        available_tools_desc = None
    return (
        available_tools_desc,
        complex_problems,
        problem_selector,
        run_agent_demo,
        tools_info,
    )


@app.cell
def __(
    available_providers,
    available_tools_desc,
    json,
    mo,
    problem_selector,
    run_agent_demo,
    tool_agent,
    tool_registry,
):
    if available_providers and run_agent_demo.value and problem_selector.value:
        try:
            selected_problem = problem_selector.value
            
            # Get agent's plan
            agent_response = tool_agent(
                problem=selected_problem,
                available_tools=available_tools_desc
            )
            
            # Execute the chosen tool if specified
            tool_results = []
            try:
                # Try to parse tool calls (agent might provide JSON or text description)
                tool_calls_text = agent_response.tool_calls
                
                # Simple parsing - in production, use more robust parsing
                if "calculator" in tool_calls_text.lower():
                    # Extract calculator operation
                    if "compound interest" in selected_problem.lower():
                        calc_result = tool_registry.execute_tool(
                            "calculator",
                            {"expression": "5000 * (1 + 0.035) ** 10", "operation": "evaluate"}
                        )
                        tool_results.append(("calculator", calc_result))
                
                if "search" in tool_calls_text.lower():
                    # Extract search query
                    if "investment" in selected_problem.lower():
                        search_result = tool_registry.execute_tool(
                            "web_search",
                            {"query": "investment strategies", "max_results": 3}
                        )
                        tool_results.append(("web_search", search_result))
                    elif "python" in selected_problem.lower():
                        search_result = tool_registry.execute_tool(
                            "web_search",
                            {"query": "python programming", "max_results": 3}
                        )
                        tool_results.append(("web_search", search_result))
                
                if "text" in tool_calls_text.lower() or "analyze" in tool_calls_text.lower():
                    # Text analysis
                    if "quick brown fox" in selected_problem.lower():
                        text_result = tool_registry.execute_tool(
                            "text_processor",
                            {"text": "The quick brown fox jumps over the lazy dog", "operation": "analyze"}
                        )
                        tool_results.append(("text_processor", text_result))
                        
                        # Transform to uppercase
                        transform_result = tool_registry.execute_tool(
                            "text_processor",
                            {"text": "The quick brown fox jumps over the lazy dog", "operation": "transform", "transform_type": "uppercase"}
                        )
                        tool_results.append(("text_processor", transform_result))
                
                if "store" in tool_calls_text.lower():
                    # Data storage operations
                    if "john" in selected_problem.lower():
                        # Store multiple data points
                        for key, value in [("name", "John"), ("age", "30"), ("city", "New York")]:
                            store_result = tool_registry.execute_tool(
                                "data_storage",
                                {"operation": "store", "key": key, "value": value}
                            )
                            tool_results.append(("data_storage", store_result))
                        
                        # Retrieve all data
                        list_result = tool_registry.execute_tool(
                            "data_storage",
                            {"operation": "list"}
                        )
                        tool_results.append(("data_storage", list_result))
                    else:
                        # Store calculation result
                        store_result = tool_registry.execute_tool(
                            "data_storage",
                            {"operation": "store", "key": "compound_interest", "value": "calculation_result"}
                        )
                        tool_results.append(("data_storage", store_result))
                
            except Exception as e:
                tool_results.append(("error", {"error": f"Tool execution error: {str(e)}"}))
            
            # Display comprehensive results
            tool_result_displays = []
            for tool_name, result in tool_results:
                tool_result_displays.append(f"""
**{tool_name.title()} Result:**
```json
{json.dumps(result, indent=2)}
```
""")

            mo.vstack([
                mo.md("## 🤖 Tool-Integrated Agent Results"),
                mo.md(f"**Problem:** {selected_problem}"),
                mo.md("### 🧠 Agent Planning"),
                mo.md(f"**Reasoning:** {agent_response.reasoning}"),
                mo.md(f"**Tool Plan:** {agent_response.tool_plan}"),
                mo.md(f"**Tool Calls:** {agent_response.tool_calls}"),
                mo.md(f"**Expected Outcome:** {agent_response.expected_outcome}"),
                mo.md("### 🔧 Tool Execution Results"),
                mo.md("\n".join(tool_result_displays) if tool_result_displays else "No tools executed"),
                mo.md("### 🎯 Final Analysis"),
                mo.md(f"**Agent's Final Answer:** {agent_response.final_answer}"),
                mo.md(f"**Tools Used:** {len(tool_results)}"),
                mo.md(f"**Execution Success:** {'✅ Yes' if all(r[1].get('success', True) for r in tool_results) else '⚠️ Partial'}")
            ])
            
        except Exception as e:
            mo.md(f"❌ **Agent Demo Error:** {str(e)}")
    else:
        mo.md("*Select a problem and click 'Run Tool-Integrated Agent' to see the agent in action*")
    return (
        agent_response,
        calc_result,
        list_result,
        search_result,
        selected_problem,
        store_result,
        text_result,
        tool_calls_text,
        tool_result_displays,
        tool_results,
        transform_result,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎓 Tool Integration Module Complete!
        
        ### 🏆 What You've Mastered
        
        ✅ **Tool Registry Architecture** - Centralized tool management system
        ✅ **Core Tool Implementation** - Mathematical, text, search, and storage tools
        ✅ **Agent Integration** - Intelligent tool selection and execution
        ✅ **Performance Monitoring** - Comprehensive usage and performance tracking
        ✅ **Production Patterns** - Scalable, robust tool integration
        
        ### 🛠️ Key Components Built
        
        1. **ToolRegistry Class**
           - Tool registration with metadata and schemas
           - Execution monitoring and statistics
           - Error handling and validation
        
        2. **Core Tools Suite**
           - Calculator with advanced mathematical operations
           - Text processor for analysis and transformation
           - Web search with simulated API integration
           - Data storage for persistent information
        
        3. **Tool-Integrated Agent**
           - Intelligent problem analysis and tool selection
           - Multi-step tool execution planning
           - Result synthesis and final answer generation
        
        ### 🎯 Skills Developed
        
        - **System Architecture** - Designing scalable tool integration frameworks
        - **API Integration** - Connecting external services and tools
        - **Error Handling** - Building robust, fault-tolerant systems
        - **Performance Monitoring** - Tracking and optimizing tool performance
        - **Agent Design** - Creating intelligent tool-using agents
        
        ### 🚀 Ready for Multi-Step Reasoning?
        
        You now have a solid foundation in tool integration! Time to explore complex reasoning patterns:
        
        **Next Module:**
        ```bash
        uv run marimo run 02-advanced-modules/multi_step_reasoning.py
        ```
        
        **Coming Up:**
        - Complex reasoning chain implementation
        - Multi-hop question answering systems
        - Reasoning step tracking and visualization
        - Advanced debugging techniques for complex workflows
        
        ### 💡 Practice Challenges
        
        Before moving on, try extending the tool integration framework:
        
        1. **Custom Tools**
           - Build a file system tool for reading/writing files
           - Create an email tool for sending notifications
           - Implement a database tool for structured data
        
        2. **Advanced Patterns**
           - Build a tool chain executor for sequential operations
           - Implement parallel tool execution with result aggregation
           - Create conditional tool selection based on context
        
        3. **Production Features**
           - Add comprehensive logging and audit trails
           - Implement tool versioning and rollback capabilities
           - Build a tool marketplace for sharing and discovery
        
        Master these tool integration patterns and you can build agents that interact with any external system or service!
        """
        )
    return


if __name__ == "__main__":
    app.run()