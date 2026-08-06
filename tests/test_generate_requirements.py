from generate_requirements import build_requirements_content


def test_build_requirements_content_includes_core_deps_and_demo_extras():
    pyproject_data = {
        "project": {
            "dependencies": ["chromadb>=0.5", "requests>=2.31"],
            "optional-dependencies": {
                "dev": ["pytest>=8.0", "ruff>=0.16,<0.17"],
                "demo": ["streamlit>=1.61", "huggingface_hub>=1.26"],
            },
        }
    }

    content = build_requirements_content(pyproject_data)

    assert "chromadb>=0.5" in content
    assert "requests>=2.31" in content
    assert "streamlit>=1.61" in content
    assert "huggingface_hub>=1.26" in content
    # dev-only extras must not leak into the deployed demo's requirements
    assert "pytest>=8.0" not in content
    assert "ruff" not in content
    # "." must appear as an exact line (a substring check would also match
    # inside version specifiers like ">=3.0", so split into lines first)
    assert "." in content.strip().split("\n")


def test_build_requirements_content_is_one_package_per_line():
    pyproject_data = {
        "project": {
            "dependencies": ["chromadb>=0.5"],
            "optional-dependencies": {"demo": ["streamlit>=1.61"]},
        }
    }

    content = build_requirements_content(pyproject_data)

    lines = content.strip().split("\n")
    assert lines == [
        "chromadb>=0.5",
        "streamlit>=1.61",
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch==2.13.0+cpu",
        ".",
    ]


def test_build_requirements_content_pins_cpu_only_torch():
    pyproject_data = {
        "project": {
            "dependencies": ["chromadb>=0.5"],
            "optional-dependencies": {"demo": ["streamlit>=1.61"]},
        }
    }

    content = build_requirements_content(pyproject_data)
    lines = content.strip().split("\n")

    # exact-line checks, not substring checks (substring would also match
    # e.g. "torch==2.13.0+cpu" inside a longer pinned specifier)
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lines
    assert "torch==2.13.0+cpu" in lines
