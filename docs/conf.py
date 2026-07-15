import os
import sys

from importlib.metadata import PackageNotFoundError, version as get_version


REPO_ROOT = os.path.abspath("..")
SRC_ROOT = os.path.join(REPO_ROOT, "src")
EXT_ROOT = os.path.join(REPO_ROOT, "docs", "_ext")

# Add the src/ root so Sphinx autodoc can import the package for docstrings,
# and docs/_ext so the local build extensions are importable by name.
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, EXT_ROOT)

# Keep the documentation build output focused. The library supports many optional
# backends and logs availability at import time; we silence those messages here.
import logging  # noqa: E402

logging.getLogger("diffract").setLevel(logging.ERROR)

project = "Diffract"
try:
    release = get_version("diffract-core")
except PackageNotFoundError:
    release = "0.2.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
    "metrics_table",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    # Generated fragment: included into catalog.md, never a standalone page.
    "reference/metrics/_generated_table.md",
]

root_doc = "index"

html_theme = "furo"
html_title = "Diffract"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "source_repository": "https://github.com/Risk-AI-Research/diffract",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["edit", "view"],
}

myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "attrs_inline",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 2
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_preserve_defaults = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True
