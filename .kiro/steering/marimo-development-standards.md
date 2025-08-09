---
inclusion: fileMatch
fileMatchPattern: '*.py'
---

# Marimo Notebook Development Standards

This document outlines the development standards and best practices for creating marimo notebooks in the DSPy learning project.

## Import Management

### Import Structure

- **Only marimo import should be outside cells**: The `import marimo` statement is the only import that should appear at the module level
- **All other imports go inside cells**: Place all other imports in the first cell or subsequent cells as needed
- **Use linting suppressions**: Include these comments at the top of every marimo notebook:

  ```python
  # pylint: disable=import-error,import-outside-toplevel,reimported
  # cSpell:ignore marimo dspy
  ```

### Example Import Pattern

```python
# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

# At module level - ONLY marimo
import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")

# Inside first cell
@app.cell
def _():
    import logging
    import sys
    from inspect import cleandoc
    from pathlib import Path
    
    import dspy
    import marimo as mo
    from marimo import output
    
    from common import (
        DSPyParameterPanel,
        SignatureBuilder,
        get_config,
    )
    
    return (
        # ... other imports
        DSPyParameterPanel,
        cleadoc,
        dspy,
        mo,
        output,
        # ... other imports
    )
```

## Text Formatting and Indentation

### Using cleandoc for Clean Text

- **Always import `cleandoc` from `inspect`** for multi-line text formatting
- **Use `cleandoc()` to provide clean indentation-issue free blob of text when required**
- **Prevents indentation parsing errors** in marimo notebooks
- **Maintains readable code structure** while ensuring proper text formatting

### Example cleandoc Usage

```python
@app.cell
def _():
    from inspect import cleandoc
    return (cleandoc,)

@app.cell
def _(cleandoc, mo, output):
    # Use cleandoc for multi-line text to avoid indentation issues
    content_text = cleandoc(
        """
        ## Welcome to DSPy Tutorial
        
        This is a comprehensive guide that covers:
        1. Basic concepts and foundations
        2. Advanced patterns and techniques
        3. Real-world applications
        
        Each section builds upon the previous one to ensure
        a complete understanding of the framework.
        """
    )
    
    cell2_out = mo.md(content_text)
    output.replace(cell2_out)
    return

# For code blocks within markdown
@app.cell
def _(cleandoc, mo, output):
    code_example = cleandoc(
        """
        ```python
        class ExampleSignature(dspy.Signature):
            \"\"\"Example signature with proper formatting\"\"\"
            
            input_field = dspy.InputField(desc="Input description")
            output_field = dspy.OutputField(desc="Output description")
        ```
        """
    )
    
    cell3_desc = mo.md("### Code Example")
    cell3_content = mo.md(code_example)
    
    cell3_out = mo.vstack([cell3_desc, cell3_content])   
    output.replace(cell3_out)
    return
```

### When to Use cleandoc

1. **Multi-line markdown content** with complex indentation
2. **Code blocks within markdown** to preserve formatting
3. **Long text descriptions** that span multiple lines
4. **Template strings** with embedded code or structured content
5. **Any text that might cause indentation parsing issues**

## Cell Output Management

### Display Rules

- **All output from each cell from `mo.md()` should be stored in `cell<n+1>_out`** where n is the cell number
- **If any cell has more than one part of the output and needs to be combined at the end then use the variables `cell<n+1>_desc` and `cell<n+1>_content` and at the end use `mo.vstack` to combine the two messages**. e.g.,

  ```python
  cell3_out = mo.vstack([cell3_desc, cell3_content])
  ```

- **Use `output.replace(cell<n+1>_out)` as the final statement before the return statement for each cell**. e.g.,

  ```python
  output.replace(cell8_out)
  return <variables if any>
  ```

- **We should never redefine a variable with the same name in a different cell. For variables with similar name use `cell<n+1>_` as the prefix**. e.g., `cell5_result`

### Example Cell Structure

```python
@app.cell
def _(mo, output):
    # Single output
    content = "Your content here"
    cell3_out = mo.md(content)
    output.replace(cell3_out)
    return  # Optional, can return variables if needed

# Multi-part output example
@app.cell
def _(mo, output):
    cell4_desc = mo.md("## Section Title")
    cell4_content = mo.md("Section content here")
    
    cell4_out = mo.vstack([cell4_desc, cell4_content])

    output.replace(cell4_out)
    return
```

## Variable Management

### Variable Naming Rules

- **We should never redefine a variable with the same name in a different cell**
- **For variables with similar name use `cell<n+1>_` as the prefix** (e.g., `cell5_result`)
- **Private variables**: For variables used only within one cell, use `_` as suffix (e.g., `_result`, `_start_time`) - these can be reused across cells
- **Mutation is allowed**: You can mutate the content of variables (e.g., dictionaries, lists)
- **Use descriptive prefixes**: Consider prefixing variables with cell identifiers for clarity

### Example Variable Management

```python
# Cell 1
@app.cell
def _():
    cell1_config = get_config()
    _temp_data = process_config(cell1_config)  # Private variable
    return (cell1_config,)

# Cell 2 - DON'T reuse 'config', use different name
@app.cell
def _(cell1_config):
    cell2_providers = cell1_config.get_available_llm_providers()
    _temp_data = validate_providers(cell2_providers)  # OK to reuse private variable
    return (cell2_providers,)

# Cell 3 - Use cell prefix for similar variables
@app.cell
def _(cell2_providers):
    cell3_result = analyze_providers(cell2_providers)
    return (cell3_result,)

# Cell 4 - Different result variable
@app.cell
def _(cell3_result):
    cell4_result = transform_result(cell3_result)
    return (cell4_result,)
```

## Conditional Rendering Anti-Patterns

### Avoid Uber Guards

**❌ Bad Pattern - Avoid This:**

```python
@app.cell
def _(available_providers, predictor, question_input, param_panel):
    if available_providers and predictor and question_input.value and param_panel:
        # All your core rendering logic wrapped in this guard
        result = complex_processing()
        output.replace(mo.md(result))
    else:
        output.replace(mo.md("Waiting..."))
    return
```

**✅ Good Pattern - Use This:**

```python
@app.cell
def _(available_providers, predictor, question_input, param_panel):
    if available_providers and predictor and question_input.value and param_panel:
        result = complex_processing()
        cell_output = mo.md(result)
    else:
        cell_output = mo.md("Waiting for input...")
    
    output.replace(cell_output)  # Always return something
    return
```

### Why Uber Guards Are Problematic

1. **Dependency Tracking Issues**: Marimo's static dependency tracking may not work correctly
2. **Reactivity Problems**: Cells may not re-run when dependencies change
3. **Stale Output**: May result in no output or stale content when conditions aren't met
4. **Execution Short-Circuiting**: Early returns can prevent proper cell execution

## Recommended Patterns

### Pattern A: Always Return Valid UI

```python
@app.cell
def _(condition1, condition2):
    if condition1 and condition2:
        content = process_data()
    else:
        content = "Please configure prerequisites"
    
    output.replace(mo.md(content))
    return
```

### Pattern B: Ternary-Style Conditional

```python
@app.cell
def _(condition):
    content = process_data() if condition else "Waiting for input..."
    output.replace(mo.md(content))
    return
```

### Pattern C: Early Content Preparation

```python
@app.cell
def _(dependencies):
    # Prepare content first
    if not dependencies:
        cell_content = mo.md("Setup required")
    else:
        result = complex_operation(dependencies)
        cell_content = mo.vstack([
            mo.md("## Results"),
            mo.md(f"Output: {result}")
        ])
    
    # Always display something
    output.replace(cell_content)
    return
```

## Error Handling

### Graceful Error Display

```python
@app.cell
def _(inputs):
    try:
        result = risky_operation(inputs)
        cell_output = mo.md(f"Success: {result}")
    except Exception as e:
        cell_output = mo.vstack([
            mo.md("### ⚠️ Error Occurred"),
            mo.md(f"Error: {str(e)}"),
            mo.md("Please check your configuration and try again.")
        ])
    
    output.replace(cell_output)
    return
```

## UI Component Best Practices

### Consistent UI Structure

- Use `mo.vstack()` for vertical layouts
- Use `mo.hstack()` for horizontal layouts
- Provide clear labels and descriptions for interactive elements
- Use consistent styling and formatting

### Interactive Elements

```python
@app.cell
def _(mo, output):
    input_field = mo.ui.text_area(
        placeholder="Enter your input...",
        label="Clear Label",
        value="Default value if appropriate"
    )
    
    ui_layout = mo.vstack([
        mo.md("### Section Title"),
        input_field
    ])
    
    output.replace(ui_layout)
    return (input_field,)
```

## Form Components and _clone Error Prevention

### The _clone Error Problem

**❌ Problematic Pattern - Causes `AttributeError: 'dict' object has no attribute '_clone'`:**

```python
@app.cell
def _(mo, output):
    # This causes _clone errors
    form = mo.ui.form({
        "name": mo.ui.text(label="Name"),
        "email": mo.ui.text(label="Email"),
        "submit": mo.ui.run_button(label="Submit")
    })
    
    cell_out = mo.vstack([
        mo.md("### Form"),
        form
    ])
    output.replace(cell_out)
    return (form,)

@app.cell
def _(form, mo, output):
    # This will cause _clone error
    if form.value and form.value["submit"]:
        data = form.value
        # Process form data...
```

**✅ Correct Pattern - Use Individual Components:**

```python
@app.cell
def _(mo, output):
    # Create individual components
    name_input = mo.ui.text(label="Name")
    email_input = mo.ui.text(label="Email")
    submit_button = mo.ui.run_button(label="Submit")
    
    # Arrange with mo.vstack instead of mo.ui.form
    form_layout = mo.vstack([
        name_input,
        email_input,
        submit_button,
    ])
    
    cell_out = mo.vstack([
        mo.md("### Form"),
        form_layout
    ])
    output.replace(cell_out)
    return (name_input, email_input, submit_button)

@app.cell
def _(name_input, email_input, submit_button, mo, output):
    # Access individual component values
    if submit_button.value:
        name_value = name_input.value
        email_value = email_input.value
        # Process individual values...
```

### Form Component Guidelines

1. **Never use `mo.ui.form()`** - It causes cloning issues with complex nested structures
2. **Use individual components** - Create separate UI components for each form field
3. **Arrange with `mo.vstack()` or `mo.hstack()`** - Use layout components instead of forms
4. **Access individual values** - Use `component.value` instead of `form.value["field"]`
5. **Return all components** - Include all individual components in the return statement

### Complex Form Example

```python
@app.cell
def _(mo, output):
    # Individual form components
    task_input = mo.ui.text_area(
        placeholder="Describe your task...",
        label="Task Description",
        rows=3
    )
    priority_dropdown = mo.ui.dropdown(
        options=["Low", "Medium", "High"],
        label="Priority",
        value="Medium"
    )
    due_date_input = mo.ui.date(label="Due Date")
    tags_multiselect = mo.ui.multiselect(
        options=["work", "personal", "urgent", "project"],
        label="Tags"
    )
    create_button = mo.ui.run_button(label="Create Task")
    
    # Layout components
    form_layout = mo.vstack([
        mo.md("### Create New Task"),
        task_input,
        priority_dropdown,
        due_date_input,
        tags_multiselect,
        create_button,
    ])
    
    cell_out = form_layout
    output.replace(cell_out)
    return (task_input, priority_dropdown, due_date_input, tags_multiselect, create_button)

@app.cell
def _(task_input, priority_dropdown, due_date_input, tags_multiselect, create_button, mo, output):
    if create_button.value:
        # Access individual component values
        task_data = {
            "description": task_input.value,
            "priority": priority_dropdown.value,
            "due_date": due_date_input.value,
            "tags": tags_multiselect.value
        }
        
        cell_out = mo.vstack([
            mo.md("### Task Created Successfully"),
            mo.md(f"**Task:** {task_data['description']}"),
            mo.md(f"**Priority:** {task_data['priority']}"),
            mo.md(f"**Due Date:** {task_data['due_date']}"),
            mo.md(f"**Tags:** {', '.join(task_data['tags']) if task_data['tags'] else 'None'}")
        ])
    else:
        cell_out = mo.md("*Fill out the form above and click 'Create Task'*")
    
    output.replace(cell_out)
    return
```

### Why This Pattern Works

1. **Avoids cloning mechanism** - Individual components don't trigger marimo's problematic form cloning
2. **Better reactivity** - Each component can update independently
3. **Clearer dependencies** - Cell dependencies are explicit and trackable
4. **Easier debugging** - Individual component states are easier to inspect
5. **More flexible** - Components can be rearranged or reused more easily

## Documentation and Comments

### Cell Documentation

- Use clear markdown headers to structure content
- Provide context for what each cell does
- Include helpful explanations for complex operations
- Use consistent emoji and formatting for visual hierarchy

### Code Comments

- Comment complex logic within cells
- Explain DSPy-specific concepts for learning purposes
- Document parameter choices and their effects

## Performance Considerations

### Efficient Reactivity

- Minimize expensive operations in frequently-updating cells
- Cache results when appropriate
- Use conditional execution wisely (but avoid uber guards)
- Consider cell dependencies when structuring code

### Resource Management

- Clean up resources when needed
- Handle API rate limits gracefully
- Provide feedback for long-running operations

## Testing and Validation

### Input Validation

- Validate user inputs before processing
- Provide clear error messages for invalid inputs
- Handle edge cases gracefully

### Environment Checks

- Verify API keys and configuration
- Check for required dependencies
- Provide helpful setup instructions when things are missing

## Common Pitfalls to Avoid

1. **Wrapping core logic in complex conditionals**
2. **Reusing variable names across cells**
3. **Not returning UI elements from cells**
4. **Placing imports outside of cells (except marimo)**
5. **Not handling errors gracefully**
6. **Creating cells that can return None/empty output**
7. **Over-nesting conditional logic**
8. **Not providing fallback content for missing dependencies**
9. **Using `mo.ui.form()` - causes `_clone` errors**
10. **Not using consistent output variable naming (`cell<n+1>_out`)**
11. **Not using `output.replace()` as the final statement before return**
12. **Mixing public and private variable naming conventions**

## Example Cell Templates

### Single Output Cell Template

```python
@app.cell
def _(dependencies, mo, output):
    # Prepare content based on conditions
    if not dependencies:
        cell5_out = mo.md("⚠️ Please configure dependencies first")
    else:
        try:
            # Main processing logic
            _result = process_with_dependencies(dependencies)  # Private variable
            cell5_out = mo.vstack([
                mo.md("### Results"),
                mo.md(f"Output: {_result}")
            ])
        except Exception as e:
            cell5_out = mo.vstack([
                mo.md("### Error"),
                mo.md(f"An error occurred: {str(e)}")
            ])
    
    output.replace(cell5_out)
    return  # Return variables if needed by other cells
```

### Multi-Part Output Cell Template

```python
@app.cell
def _(dependencies, cleandoc, mo, output):
    if not dependencies:
        cell6_desc = mo.md("## Configuration Required")
        cell6_content = mo.md("⚠️ Please configure dependencies first")
    else:
        try:
            # Main processing logic
            _result = process_with_dependencies(dependencies)  # Private variable
            cell6_desc = mo.md("## Processing Results")
            
            # Use cleandoc for clean multi-line content
            content_text = cleandoc(
                f"""
                ### Analysis Complete
                
                **Result:** {_result}
                
                ### Next Steps
                
                Review the results above and proceed to the next section.
                Additional processing can be performed based on these results.
                """
            )
            cell6_content = mo.md(content_text)
        except Exception as e:
            cell6_desc = mo.md("## Error Occurred")
            cell6_content = mo.md(f"An error occurred: {str(e)}")
    
    cell6_out = mo.vstack([
        cell6_desc,
        cell6_content,
    ])
    
    output.replace(cell6_out)
    return  # Return variables if needed by other cells
```

### Form Components Cell Template (Avoiding _clone Errors)

```python
@app.cell
def _(mo, output):
    # Create individual components (NOT mo.ui.form)
    cell7_name_input = mo.ui.text(label="Name", placeholder="Enter your name")
    cell7_email_input = mo.ui.text(label="Email", placeholder="Enter your email")
    cell7_submit_button = mo.ui.run_button(label="Submit Form")
    
    # Arrange components with mo.vstack
    _form_layout = mo.vstack([
        cell7_name_input,
        cell7_email_input,
        cell7_submit_button,
    ])
    
    cell7_desc = mo.md("## User Information Form")
    cell7_content = _form_layout
    
    cell7_out = mo.vstack([
        cell7_desc,
        cell7_content,
    ])
    
    output.replace(cell7_out)
    return (cell7_name_input, cell7_email_input, cell7_submit_button)

@app.cell
def _(cell7_name_input, cell7_email_input, cell7_submit_button, mo, output):
    if cell7_submit_button.value:
        # Access individual component values
        _name_value = cell7_name_input.value or "Not provided"  # Private variable
        _email_value = cell7_email_input.value or "Not provided"  # Private variable
        
        cell8_out = mo.vstack([
            mo.md("### Form Submitted Successfully"),
            mo.md(f"**Name:** {_name_value}"),
            mo.md(f"**Email:** {_email_value}")
        ])
    else:
        cell8_out = mo.md("*Fill out the form above and click 'Submit Form'*")
    
    output.replace(cell8_out)
    return
```

### Key Standards These Templates Follow

1. **All output from each cell from `mo.md()` should be stored in `cell<n+1>_out`**
2. **If any cell has more than one part of the output and needs to be combined at the end then use the variables `cell<n+1>_desc` and `cell<n+1>_content` and at the end use `mo.vstack` to combine the two messages**
3. **Use `output.replace(cell<n+1>_out)` as the final statement before the return statement for each cell**
4. **We should never redefine a variable with the same name in a different cell. For variables with similar name use `cell<n+1>_` as the prefix** (e.g., `cell5_result`)
5. **Private variables**: `_result`, `_name_value` for cell-internal use - these can be reused across cells
6. **Use `cleandoc` from `inspect` to provide clean indentation-issue free blob of text when required**
7. **No form components**: Individual UI components with `mo.vstack()` layout to avoid `_clone` errors
8. **Error handling**: Graceful error display with consistent formatting
9. **Conditional content**: Always return valid UI content
