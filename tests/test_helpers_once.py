"""Percentile, hardware, and git helpers exist once in this package."""

from pathlib import Path

_ROOT = Path(__file__).parents[1] / "src" / "asrbench"


def test_the_three_helpers_are_defined_once() -> None:
    texts = [path.read_text(encoding="utf-8") for path in _ROOT.rglob("*.py")]
    joined = "\n".join(texts)
    assert joined.count("def percentile(") == 1
    assert joined.count("def collect_hardware(") == 1
    assert joined.count("def git_commit(") == 1
