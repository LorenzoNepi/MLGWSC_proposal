# pylint: disable=invalid-name
# For the next time: USE A LOWER CASE NAME !!
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MLGWSC_proposal")
except PackageNotFoundError:
    __version__ = "unknown"         # Fallback in case the package was still not installed in venv
