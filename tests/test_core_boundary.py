"""The measurement core must stay runnable with nothing installed.

A benchmark others cannot run is a claim, not a measurement. So the core —
schemas, runner, reducers, reporters — imports only the standard library, and
a transport's client library is confined to the adapter that speaks that
protocol. A misplaced import would not fail in a developer environment where
everything is installed; it would fail for the first outside user, long after
the coupling was introduced. Hence a scan rather than a discovery.

The permitted set is read from ``pyproject.toml`` rather than restated here,
so declaring a new extra is the single act that widens the boundary.
"""

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib

_ROOT = Path(__file__).parents[1]
_PACKAGE = _ROOT / "src" / "asrbench"

#: Extras that exist to carry a transport client, mapped to the package
#: subtree permitted to import them. Anything outside that subtree importing
#: the library would make the whole framework require it.
_TRANSPORT_EXTRAS = {
    "websocket": Path("adapters"),
    "riva": Path("adapters"),
}

#: Distribution name -> importable module, where they differ.
_IMPORT_NAME = {"nvidia-riva-client": "riva", "grpcio": "grpc"}


def _pyproject() -> dict[str, Any]:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _transport_modules() -> dict[str, Path]:
    """Return importable module -> the subtree allowed to import it."""
    extras = _pyproject()["project"]["optional-dependencies"]
    modules: dict[str, Path] = {}
    for extra, subtree in _TRANSPORT_EXTRAS.items():
        for requirement in extras[extra]:
            name = str(requirement).split(">")[0].split("=")[0].strip()
            modules[_IMPORT_NAME.get(name, name.replace("-", "_"))] = subtree
    return modules


def _imports(path: Path) -> list[tuple[str, int]]:
    """Return every (top-level module, line) this file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                (alias.name.split(".")[0], node.lineno) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # intra-package relative import
                continue
            if node.module:
                found.append((node.module.split(".")[0], node.lineno))
    return found


def _sources() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def test_the_package_has_sources_to_scan() -> None:
    """A scan over an empty tree would pass while proving nothing."""
    assert _sources()


def test_the_core_declares_no_runtime_dependencies() -> None:
    """Installing the framework must not install anything."""
    assert _pyproject()["project"]["dependencies"] == []


@pytest.mark.parametrize(
    "source", _sources(), ids=lambda p: str(p.relative_to(_PACKAGE))
)
def test_only_the_stdlib_and_declared_transports_are_imported(
    source: Path,
) -> None:
    """An undeclared dependency is a dependency an installer will not get."""
    allowed = (
        set(sys.stdlib_module_names) | set(_transport_modules()) | {"asrbench"}
    )

    offenders = [
        f"{source.relative_to(_PACKAGE)}:{line} imports {module!r}"
        for module, line in _imports(source)
        if module not in allowed
    ]

    assert not offenders, (
        "the core is stdlib-only; declare a transport extra in pyproject.toml "
        f"before importing it. Offending imports: {offenders}"
    )


@pytest.mark.parametrize(
    "source", _sources(), ids=lambda p: str(p.relative_to(_PACKAGE))
)
def test_transport_libraries_stay_inside_their_adapter(source: Path) -> None:
    """A missing extra disables one transport, never the framework."""
    relative = source.relative_to(_PACKAGE)
    scoped = _transport_modules()

    offenders = [
        f"{relative}:{line} imports {module!r}"
        for module, line in _imports(source)
        if module in scoped and scoped[module] not in relative.parents
    ]

    assert not offenders, (
        "a transport library belongs to its own adapter subtree; "
        f"misplaced: {offenders}"
    )


def test_the_scan_would_catch_a_real_escape(tmp_path: Path) -> None:
    """The boundary check must fail on a violation, not merely pass."""
    escaped = tmp_path / "escaped.py"
    escaped.write_text("import torch\nfrom vllm import LLM\n", encoding="utf-8")

    modules = {module for module, _ in _imports(escaped)}
    allowed = set(sys.stdlib_module_names) | set(_transport_modules())

    assert modules == {"torch", "vllm"}
    assert not modules & allowed
