import os
import sys

from importlib.metadata import PackageNotFoundError, version as get_version


REPO_ROOT = os.path.abspath("..")
SRC_ROOT = os.path.join(REPO_ROOT, "src")

# Add the src/ root so Sphinx autodoc can import the package for docstrings.
sys.path.insert(0, SRC_ROOT)

# Keep the documentation build output focused. The library supports many optional
# backends and logs availability at import time; we silence those messages here.
import logging  # noqa: E402

logging.getLogger("diffract").setLevel(logging.ERROR)

project = "Diffract"
try:
    release = get_version("diffract")
except PackageNotFoundError:
    release = "0.1.0"
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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

root_doc = "index"

html_theme = "furo"
html_title = "Diffract"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Removed for anonymized submission
# html_theme_options = {
#     "source_repository": "https://github.com/...",
#     "source_branch": "main",
#     "source_directory": "docs/",
#     "navigation_with_keys": True,
#     "top_of_page_buttons": ["edit", "view"],
# }

myst_enable_extensions = ["deflist", "colon_fence", "attrs_inline"]
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
