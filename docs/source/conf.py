"""
Sphinx configuration for welltest-pta documentation.

This file is executed by Sphinx with the current directory set to its
containing directory (docs/source). See https://www.sphinx-doc.org/.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — make the package importable
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import welltest_pta  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Project information
# ─────────────────────────────────────────────────────────────────────────────
project = "welltest-pta"
author = "Ismail Harkat (geoharkat)"
copyright = f"{datetime.now():%Y}, {author}"

# The short X.Y version
version = ".".join(welltest_pta.__version__.split(".")[:2])
# The full version, including alpha/beta/rc tags
release = welltest_pta.__version__

# ─────────────────────────────────────────────────────────────────────────────
# General configuration
# ─────────────────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# MyST: enable a few extensions for Markdown imports (e.g. CHANGELOG.md)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress duplicate-object-description warnings — these arise because
# autosummary generates stub pages for top-level re-exports while the
# canonical descriptions live in the submodule pages. The build still
# succeeds; the warnings are noise only.
suppress_warnings: list[str] = []

# ─────────────────────────────────────────────────────────────────────────────
# Autodoc / Napoleon
# ─────────────────────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__,__call__",
    "undoc-members": False,
    "exclude-members": "__weakref__,__dict__,__module__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True
autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# sphinx-autodoc-typehints
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# ─────────────────────────────────────────────────────────────────────────────
# Intersphinx — cross-link Python / NumPy / SciPy / Pandas / Matplotlib
# ─────────────────────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 30

# ─────────────────────────────────────────────────────────────────────────────
# HTML output
# ─────────────────────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
html_static_path = ["_static"]
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_title = f"{project} v{release}"
html_short_title = project

# Favicon & logo (uncomment when assets exist)
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX / PDF output
# ─────────────────────────────────────────────────────────────────────────────
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{booktabs}
    """,
}
latex_documents = [(
    master_doc,
    "welltest-pta.tex",
    "welltest-pta Documentation",
    author,
    "manual",
)]

# ─────────────────────────────────────────────────────────────────────────────
# sphinx-copybutton
# ─────────────────────────────────────────────────────────────────────────────
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────
pygments_style = "sphinx"
todo_include_todos = True
nitpicky = False
