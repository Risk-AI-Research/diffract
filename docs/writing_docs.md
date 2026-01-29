# Writing Documentation

This guide explains how to add and update documentation for Diffract.

## Structure

The documentation follows a book-first structure:

- **`guide/`** - Tutorials and conceptual guides
- **`examples/`** - Example-driven pages
- **`reference/`** - API reference generated from docstrings

## Adding Guide Pages

1. Create a new `.md` file in `docs/guide/` or `docs/guide/recipes/`
2. Add it to the appropriate `toctree` in `docs/index.md`
3. Write in MyST Markdown (supports all standard Markdown + Sphinx directives)

Example:

```markdown
# My New Guide

This is a guide about X.

## Section

Some content here.
```

## Adding API Documentation

API docs are generated from docstrings. To document a class or function:

1. **Write Google-style docstrings** in your Python code:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description.

    Longer description explaining what the function does.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something goes wrong.
    """
    ...
```

2. **Add it to the reference** by creating or updating a file in `docs/reference/`:

```markdown
# Session module

```{eval-rst}
.. automodule:: diffract.session
   :members:
   :undoc-members:
   :show-inheritance:
```
```

Or for a specific class:

```markdown
# Session

```{eval-rst}
.. autoclass:: diffract.session.Session
   :members:
   :show-inheritance:
```
```

**Important:** Use `eval-rst` directive when using RST autodoc directives in Markdown files.

## Building and Viewing

```bash
# Build documentation
make docs

# View in browser
open docs/_build/html/index.html

# Clean build artifacts
make docs-clean
```

## MyST Markdown Features

You can use:

- Standard Markdown syntax
- Code blocks with syntax highlighting
- Sphinx directives via `{directive}` syntax
- RST directives via `{eval-rst}` blocks
- Grid layouts with `sphinx-design`:

```markdown
```{grid} 2
:gutter: 2

```{grid-item-card} Card Title
:link: some-page
:link-type: doc

Card content.
```
```

## Tips

- Keep docstrings concise but informative
- Use type hints - they appear automatically in the docs
- Add examples in docstrings when helpful
- Update the `toctree` in `docs/index.md` when adding new top-level pages
