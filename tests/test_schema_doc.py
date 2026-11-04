"""The checked-in schema reference is a fresh generation."""

from pathlib import Path

from asrbench.definitions import SCHEMA_VERSION, emitted_definitions
from asrbench.schema_doc import schema_markdown

_DOC = Path(__file__).parents[1] / "docs" / "receipt-schema.md"


def test_receipt_schema_matches_a_fresh_generation() -> None:
    generated = schema_markdown()
    assert _DOC.read_text(encoding="utf-8") == generated
    assert SCHEMA_VERSION in generated
    for value in emitted_definitions().values():
        assert value in generated
