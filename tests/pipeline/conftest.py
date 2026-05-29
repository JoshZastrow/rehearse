import sys
from unittest.mock import MagicMock

# Modal is installed in the uv venv. Provide a fallback mock so tests run
# under any Python without requiring modal to be importable.
if "modal" not in sys.modules:
    sys.modules["modal"] = MagicMock()
