# pylint: disable=invalid-name
# For the next time: USE A LOWER CASE NAME !!
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MLGWSC_proposal")
except PackageNotFoundError:
    __version__ = "unknown"         # Fallback in case the package was still not installed in venv

"""
# Previews versioning method

import pathlib
import subprocess

from ._version import __version__ as __base_version__

def _git_suffix():
    # Function to retrieve git suffix and add '.dirty' if uncommitted changes are present.
    cwd = pathlib.Path(__file__).parent

    try:
        # Get the commit SHA
        sha_process = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], # git command
            cwd=cwd,
            capture_output=True,    # "Captures" the outputs
            text=True,              # Converts SHA from binary to text
            check=True              # Raises subprocess.CalledProcessError if run fails
        )
        sha = sha_process.stdout.strip()
        suffix = f"+g{sha}"

        # Check for uncommitted changes for the '.dirty' tag
        diff_process = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=False
        )

        if diff_process.returncode != 0:
            suffix = f"{suffix}.dirty"

        return suffix

    except FileNotFoundError:   # i.e. git not installed
        return ""
    except subprocess.CalledProcessError:   # subprocess.run(...) failed
        return ""

__version__ = f"{__base_version__}{_git_suffix()}"
"""
