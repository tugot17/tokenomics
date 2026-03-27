__version__ = "0.5.5"

from pathlib import Path

# Bundled in wheel as tokenomics/examples/, in dev at repo root examples/
_pkg_examples = Path(__file__).parent / "examples"
_repo_examples = Path(__file__).parent.parent / "examples"
EXAMPLES_DIR = _pkg_examples if _pkg_examples.exists() else _repo_examples
