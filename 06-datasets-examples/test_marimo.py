# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore marimo dspy

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from marimo import output

    return mo, output


@app.cell
def _(mo, output):
    cell2_out = mo.md("# Test Notebook")
    output.replace(cell2_out)
    return


if __name__ == "__main__":
    app.run()
