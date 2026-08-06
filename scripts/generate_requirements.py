# scripts/generate_requirements.py
"""Generates requirements.txt from pyproject.toml's core dependencies plus
the `demo` optional-dependencies group, so Streamlit Community Cloud (which
checks requirements.txt before pyproject.toml, and only supports Poetry-
style pyproject.toml dependency blocks -- this project uses plain PEP 621
[project.dependencies], not Poetry) has a dependency file it can use
without relying on untested parser compatibility.

pyproject.toml stays the single source of truth. Run this script after any
dependency change and commit the result; `--check` (used by CI) fails
instead of writing if the committed file doesn't match what this script
would currently generate.

Usage: python scripts/generate_requirements.py [--check]
"""
import argparse
import sys
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
REQUIREMENTS_PATH = Path("requirements.txt")


def build_requirements_content(pyproject_data: dict) -> str:
    """Pure function: takes already-parsed pyproject.toml data (as
    tomllib.load/tomllib.loads would produce) and returns the exact
    requirements.txt text -- core project.dependencies plus
    project.optional-dependencies.demo, one package specifier per line.
    No file IO, no argv -- testable in isolation from the CLI wrapper."""
    core_deps = pyproject_data["project"]["dependencies"]
    demo_deps = pyproject_data["project"]["optional-dependencies"]["demo"]
    # sentence-transformers (a core dep) pulls in torch transitively. On Linux
    # deploy targets (e.g. Streamlit Community Cloud), a plain `pip install
    # torch` resolves to the CUDA-enabled build by default, dragging in
    # several GB of nvidia-* wheels and commonly blowing the build step's
    # resource/time budget. Pin an explicit CPU-only torch build so pip never
    # considers the CUDA wheel.
    torch_cpu_pin = [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch==2.13.0+cpu",
    ]
    # Self-install this project (equivalent to `pip install .`) so that
    # installing from requirements.txt alone -- which is all Streamlit
    # Community Cloud does; it never runs `pip install -e .` / `pip install .`
    # against the repo -- also installs the fda_device_rag package itself.
    self_install = ["."]
    return "\n".join(core_deps + demo_deps + torch_cpu_pin + self_install) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pyproject_data = tomllib.loads(PYPROJECT_PATH.read_text())
    content = build_requirements_content(pyproject_data)

    if args.check:
        existing = REQUIREMENTS_PATH.read_text() if REQUIREMENTS_PATH.exists() else ""
        if existing != content:
            print(f"{REQUIREMENTS_PATH} is out of date with {PYPROJECT_PATH} -- "
                  f"run 'python scripts/generate_requirements.py' and commit the result.")
            sys.exit(1)
        print(f"{REQUIREMENTS_PATH} matches {PYPROJECT_PATH}.")
    else:
        REQUIREMENTS_PATH.write_text(content)
        print(f"Wrote {REQUIREMENTS_PATH}")


if __name__ == "__main__":
    main()
