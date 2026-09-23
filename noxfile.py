import pathlib
import shutil

import nox

from MLGWSC_proposal import __name__ as __package_name__

# Basic environment.
_ROOT_DIR = pathlib.Path(__file__).parent                           # Root directory of the project
_DOCS_DIR = _ROOT_DIR / "docs"                                      # Documentation directory
_SRC_DIR = _ROOT_DIR / "src" / __package_name__                     # Source code directory
_TESTS_DIR = _ROOT_DIR / "tests"                                    # Unit test directory

# Folders containing code that potentially needs linting
_LINT_DIRS = ("src", "tests")

# Reuse existing virtualenvs by default
nox.options.reuse_existing_virtualenvs = True

# Session to cleanup pycache and other temporary files --> nox -s cleanup
@nox.session(venv_backend="none")
def cleanup(session: nox.Session) -> None:
    """Cleanup temporary files in all directories and sub-directories."""
    cache_dirs = [
        *_ROOT_DIR.rglob("__pycache__"),
        *_ROOT_DIR.rglob(".mypy_cache"),
        *_ROOT_DIR.rglob(".ruff_cache"),
        *_ROOT_DIR.rglob(".pytest_cache"),
        _ROOT_DIR / ".nox",
        _DOCS_DIR / "build"
    ]

    for path in cache_dirs:
        shutil.rmtree(path, ignore_errors=True)

# Session to create the documentation locally --> nox -s docs
@nox.session(venv_backend="none")
def docs(session: nox.Session) -> None:
    """Generate docs."""
    source_dir = _DOCS_DIR / "source"
    output_dir = _DOCS_DIR / "build" / "html"
    session.run("sphinx-build", "-b", "html", source_dir, output_dir, *session.posargs)

# Session to run ruff on the necessary files --> nox -s ruff
@nox.session
def ruff(session: nox.Session) -> None:
    """Run ruff."""
    session.install("ruff")
    session.run("ruff", "check", *session.posargs)

# Session to run pylint --> nox -s pylint
@nox.session
def pylint(session: nox.Session) -> None:
    """Run pylint."""
    session.install("pylint")
    session.install(".[dev]")
    session.run("pylint", *_LINT_DIRS, *session.posargs)

# Session to run mypy --> nox -s mypy
@nox.session
def mypy(session: nox.Session) -> None:
    """Run mypy."""
    session.install("mypy")
    session.install(".[dev]")
    session.run("mypy", *_LINT_DIRS, *session.posargs )

# Session to run unit tests --> nox -s test
@nox.session
def test(session: nox.Session) -> None:
    """Run the unit tests.
    """
    session.install("pytest")
    session.install(".[dev]")
    session.run("pytest", *session.posargs)
