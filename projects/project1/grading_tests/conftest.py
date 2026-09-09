"""Local submission loading adapted from refs/projects/project1/grading_tests."""

import importlib.util
import pathlib
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--submission-dir",
        default=".",
        help="Local directory containing implementations.py and README.md.",
    )


@pytest.fixture(scope="session")
def github_repo_path(pytestconfig) -> pathlib.Path:
    # Keep the fixture name used by the existing public tests.
    return pathlib.Path(pytestconfig.getoption("--submission-dir")).resolve()


@pytest.fixture(scope="session")
def student_implementations(github_repo_path: pathlib.Path):
    source = github_repo_path / "implementations.py"
    if not source.is_file():
        pytest.fail(f"Missing file {source}; check --submission-dir.")
    spec = importlib.util.spec_from_file_location("student_implementations", source)
    handle = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(github_repo_path))
        patch.setitem(sys.modules, spec.name, handle)
        spec.loader.exec_module(handle)
        yield handle
