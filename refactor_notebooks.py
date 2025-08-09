#!/usr/bin/env python3
"""
Refactor marimo notebooks to follow development standards.

This script implements the marimo notebook refactoring system as specified
in .kiro/specs/marimo-notebook-refactoring/. It processes notebooks in the
01-foundations, 02-advanced-modules, and 03-retrieval-rag directories.

Usage:
    uv run refactor_notebooks.py [--dry-run] [--verbose] [--backup]
"""

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RefactoringResult:
    """Result of refactoring a single notebook."""

    file_path: Path
    success: bool
    transformations_applied: List[str]
    issues_found: List[str]
    issues_fixed: List[str]
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class RefactoringConfig:
    """Configuration for the refactoring process."""

    target_directories: List[Path]
    backup_enabled: bool = True
    dry_run: bool = False
    verbose_logging: bool = False


class MarimoCellExtractor:
    """Extract and manipulate marimo cells."""

    @staticmethod
    def extract_cells(content: str) -> List[Dict[str, str]]:
        """Extract individual cells from notebook content."""
        cell_pattern = r"(@app\.cell\s*\ndef\s+[^:]+:.*?)(?=@app\.cell|\nif __name__|$)"
        cells = []

        for match in re.finditer(cell_pattern, content, re.DOTALL):
            cell_content = match.group(1)
            cells.append(
                {"content": cell_content, "start": match.start(), "end": match.end()}
            )

        return cells

    @staticmethod
    def reconstruct_content(original_content: str, cells: List[Dict[str, str]]) -> str:
        """Reconstruct notebook content from modified cells."""
        # Find the part before first cell and after last cell
        first_cell_start = (
            min(cell["start"] for cell in cells) if cells else len(original_content)
        )
        last_cell_end = max(cell["end"] for cell in cells) if cells else 0

        header = original_content[:first_cell_start]
        footer = original_content[last_cell_end:]

        # Combine header + cells + footer
        cell_contents = [cell["content"] for cell in cells]
        return header + "\n".join(cell_contents) + footer


def apply_import_fixes(content: str, verbose: bool = False) -> Tuple[str, List[str]]:
    """Apply import-related fixes."""
    fixes_applied = []

    # Add linting suppressions if not present
    if not content.startswith("# pylint: disable="):
        suppressions = "# pylint: disable=import-error,import-outside-toplevel,reimported\n# cSpell:ignore marimo dspy\n\n"
        content = suppressions + content
        fixes_applied.append("Added linting suppressions")
        if verbose:
            print("  - Added linting suppressions")

    # Fix first cell imports - add output import
    first_cell_pattern = r"(@app\.cell\s*\ndef\s+[^:]+:\s*\n)(.*?)(return\s*\([^)]*\))"

    def fix_first_cell(match):
        cell_def = match.group(1)
        cell_body = match.group(2)
        return_stmt = match.group(3)

        # Add output import if not present
        if "from marimo import output" not in cell_body:
            # Find the marimo import line and add output import after it
            if "import marimo as mo" in cell_body:
                cell_body = cell_body.replace(
                    "import marimo as mo",
                    "import marimo as mo\n    from marimo import output",
                )
                fixes_applied.append("Added output import")
                if verbose:
                    print("  - Added output import")

        # Update return statement to include output
        if "output," not in return_stmt and "output" not in return_stmt:
            return_stmt = return_stmt.replace("return (", "return (\n        output,")
            fixes_applied.append("Updated return statement to include output")
            if verbose:
                print("  - Updated return statement to include output")

        return cell_def + cell_body + return_stmt

    content = re.sub(first_cell_pattern, fix_first_cell, content, flags=re.DOTALL)
    return content, fixes_applied


def apply_cell_output_fixes(
    content: str, verbose: bool = False
) -> Tuple[str, List[str]]:
    """Apply cell output-related fixes."""
    fixes_applied = []

    # Pattern to match cells that end with mo.md() or UI elements without output.replace()
    cells = MarimoCellExtractor.extract_cells(content)
    modified_cells = []

    for cell in cells:
        cell_content = cell["content"]
        original_content = cell_content

        # Check if cell has display content but no output.replace
        if (
            "mo.md(" in cell_content
            or "mo.vstack(" in cell_content
            or "mo.hstack(" in cell_content
        ) and "output.replace(" not in cell_content:

            # Look for the last expression that should be wrapped
            lines = cell_content.split("\n")
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if (
                    line.startswith("mo.md(")
                    or line.startswith("mo.vstack(")
                    or line.startswith("mo.hstack(")
                    or line.endswith("_selector")
                    or line.endswith("_button")
                ):

                    # Check if output is available in this cell
                    if "output" in cell_content and "def _(" in cell_content:
                        # Wrap the expression with output.replace
                        if not line.startswith("output.replace("):
                            lines[i] = f"    output.replace({line})"
                            fixes_applied.append(f"Added output.replace() wrapper")
                            if verbose:
                                print(f"  - Added output.replace() wrapper")
                    break

            cell_content = "\n".join(lines)

        cell["content"] = cell_content
        modified_cells.append(cell)

    if fixes_applied:
        content = MarimoCellExtractor.reconstruct_content(content, modified_cells)

    return content, fixes_applied


def apply_variable_naming_fixes(
    content: str, verbose: bool = False
) -> Tuple[str, List[str]]:
    """Apply variable naming fixes to avoid reuse across cells."""
    fixes_applied = []

    # Extract cells and track variable usage
    cells = MarimoCellExtractor.extract_cells(content)
    common_vars = ["available_providers", "config", "result", "response"]

    for i, cell in enumerate(cells):
        cell_content = cell["content"]
        original_content = cell_content

        # Replace common variables with cell-specific names
        for var in common_vars:
            if f"{var} =" in cell_content and f"cell{i+1}_{var}" not in cell_content:
                cell_content = cell_content.replace(f"{var} =", f"cell{i+1}_{var} =")
                cell_content = cell_content.replace(
                    f"return {var}", f"return cell{i+1}_{var}"
                )
                cell_content = cell_content.replace(
                    f"return ({var}", f"return (cell{i+1}_{var}"
                )
                fixes_applied.append(f"Renamed {var} to cell{i+1}_{var}")
                if verbose:
                    print(f"  - Renamed {var} to cell{i+1}_{var}")

        cell["content"] = cell_content

    if fixes_applied:
        content = MarimoCellExtractor.reconstruct_content(content, cells)

    return content, fixes_applied


def apply_conditional_rendering_fixes(
    content: str, verbose: bool = False
) -> Tuple[str, List[str]]:
    """Apply conditional rendering fixes to avoid uber guards."""
    fixes_applied = []

    # Pattern to find complex conditional guards that wrap rendering logic
    uber_guard_pattern = r"(if\s+\w+.*?and.*?and.*?:)\s*\n(.*?mo\.(?:md|vstack|hstack).*?)\n(\s*else:\s*\n.*?)(?=\n\s*return|\n@app\.cell|\Z)"

    def fix_uber_guard(match):
        condition = match.group(1)
        if_body = match.group(2)
        else_body = match.group(3)

        # Refactor to prepare content first, then display
        fixes_applied.append("Refactored uber guard to use content preparation pattern")
        if verbose:
            print("  - Refactored uber guard to use content preparation pattern")

        return f"""{condition}
        cell_content = {if_body.strip()}
    else:
        cell_content = mo.md("")
    
    output.replace(cell_content)"""

    content = re.sub(
        uber_guard_pattern, fix_uber_guard, content, flags=re.DOTALL | re.MULTILINE
    )

    return content, fixes_applied


def refactor_notebook(file_path: Path, config: RefactoringConfig) -> RefactoringResult:
    """Refactor a single marimo notebook file."""
    start_time = time.time()
    transformations_applied = []
    issues_found = []
    issues_fixed = []

    try:
        if config.verbose_logging:
            print(f"🔍 Processing: {file_path}")

        # Read original content
        original_content = file_path.read_text(encoding="utf-8")
        content = original_content

        # Create backup if enabled
        if config.backup_enabled and not config.dry_run:
            backup_path = file_path.with_suffix(".py.backup")
            shutil.copy2(file_path, backup_path)
            if config.verbose_logging:
                print(f"📁 Backup created: {backup_path}")

        # Apply refactoring patterns
        content, import_issues = apply_import_fixes(content, config.verbose_logging)
        if import_issues:
            transformations_applied.extend(import_issues)
            issues_fixed.extend(import_issues)

        content, output_issues = apply_cell_output_fixes(
            content, config.verbose_logging
        )
        if output_issues:
            transformations_applied.extend(output_issues)
            issues_fixed.extend(output_issues)

        content, variable_issues = apply_variable_naming_fixes(
            content, config.verbose_logging
        )
        if variable_issues:
            transformations_applied.extend(variable_issues)
            issues_fixed.extend(variable_issues)

        content, conditional_issues = apply_conditional_rendering_fixes(
            content, config.verbose_logging
        )
        if conditional_issues:
            transformations_applied.extend(conditional_issues)
            issues_fixed.extend(conditional_issues)

        # Write refactored content if not dry run
        if not config.dry_run:
            file_path.write_text(content, encoding="utf-8")

        execution_time = time.time() - start_time

        if config.verbose_logging or transformations_applied:
            status = "✅ Refactored" if not config.dry_run else "🔍 Would refactor"
            print(f"{status}: {file_path}")
            if transformations_applied:
                for transform in transformations_applied:
                    print(f"  - {transform}")

        return RefactoringResult(
            file_path=file_path,
            success=True,
            transformations_applied=transformations_applied,
            issues_found=issues_found,
            issues_fixed=issues_fixed,
            execution_time=execution_time,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error refactoring {file_path}: {e}"
        print(f"❌ {error_msg}")

        return RefactoringResult(
            file_path=file_path,
            success=False,
            transformations_applied=[],
            issues_found=[],
            issues_fixed=[],
            execution_time=execution_time,
            error_message=str(e),
        )


def get_notebook_files(directories: List[Path]) -> List[Path]:
    """Get all marimo notebook files to refactor."""
    notebook_files = []

    for dir_path in directories:
        if dir_path.exists() and dir_path.is_dir():
            py_files = list(dir_path.glob("*.py"))
            # Filter out non-notebook files
            for py_file in py_files:
                try:
                    content = py_file.read_text(encoding="utf-8")
                    if "import marimo" in content and "@app.cell" in content:
                        notebook_files.append(py_file)
                except Exception:
                    continue  # Skip files that can't be read
        else:
            print(f"⚠️ Directory not found: {dir_path}")

    return notebook_files


def generate_summary_report(
    results: List[RefactoringResult], config: RefactoringConfig
):
    """Generate a comprehensive summary report."""
    total_files = len(results)
    successful_files = sum(1 for r in results if r.success)
    failed_files = total_files - successful_files

    total_transformations = sum(len(r.transformations_applied) for r in results)
    total_execution_time = sum(r.execution_time for r in results)

    print("\n" + "=" * 60)
    print("📊 REFACTORING SUMMARY REPORT")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if config.dry_run else 'LIVE EXECUTION'}")
    print(f"Total files processed: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Total transformations applied: {total_transformations}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Average time per file: {total_execution_time/total_files:.2f} seconds")

    if failed_files > 0:
        print(f"\n❌ FAILED FILES:")
        for result in results:
            if not result.success:
                print(f"  - {result.file_path}: {result.error_message}")

    # Group transformations by type
    transformation_counts = {}
    for result in results:
        for transform in result.transformations_applied:
            transformation_counts[transform] = (
                transformation_counts.get(transform, 0) + 1
            )

    if transformation_counts:
        print(f"\n🔧 TRANSFORMATIONS APPLIED:")
        for transform, count in sorted(transformation_counts.items()):
            print(f"  - {transform}: {count} times")

    print("=" * 60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Refactor marimo notebooks to follow development standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run refactor_notebooks.py                    # Refactor all notebooks
  uv run refactor_notebooks.py --dry-run          # Preview changes only
  uv run refactor_notebooks.py --verbose          # Detailed logging
  uv run refactor_notebooks.py --no-backup        # Skip backup creation
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup files"
    )

    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["01-foundations", "02-advanced-modules", "03-retrieval-rag"],
        help="Directories to process (default: all three main directories)",
    )

    return parser.parse_args()


def main():
    """Main refactoring function."""
    args = parse_arguments()

    print("🔧 Starting marimo notebook refactoring...")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE EXECUTION'}")

    # Setup configuration
    target_directories = [Path(d) for d in args.dirs]
    config = RefactoringConfig(
        target_directories=target_directories,
        backup_enabled=not args.no_backup,
        dry_run=args.dry_run,
        verbose_logging=args.verbose,
    )

    # Get notebook files
    notebook_files = get_notebook_files(target_directories)
    print(f"Found {len(notebook_files)} notebook files to process")

    if not notebook_files:
        print("❌ No notebook files found to process")
        sys.exit(1)

    # Process each notebook
    results = []
    for file_path in notebook_files:
        result = refactor_notebook(file_path, config)
        results.append(result)

    # Generate summary report
    generate_summary_report(results, config)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        print(f"\n⚠️ {failed_count} files failed to process")
        sys.exit(1)
    else:
        print(f"\n✅ All files processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
