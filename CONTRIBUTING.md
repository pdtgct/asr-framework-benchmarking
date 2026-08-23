# Contributing

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Gates

The same three CI runs, on Python 3.10 and 3.12:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

`mypy` runs in strict mode over `src` and `tests`. New modules are expected to type
cleanly rather than to accumulate ignores.

## Invariants

Four rules, each protecting something a reviewer cannot easily re-derive.

**The core stays stdlib-only.** Schemas, runner, reducers, and reporters import
nothing outside the standard library. A transport's client library is confined to the
adapter that speaks that protocol, declared as an extra in `pyproject.toml`. This is
enforced by `tests/test_core_boundary.py`, not by review — so widening the boundary
means editing the extras, which is the decision you wanted to make consciously.

**The default test tier needs no GPU and no server.** If a change makes the mock path
unrunnable, the framework has stopped being publicly reproducible. Tests requiring a
live endpoint carry `@pytest.mark.live` and are excluded by default.

**Metric definitions live in code, not prose.** Every metric carries its clock and
its definition as an emitted constant, the receipt writes it, and the documentation
quotes the constant. A definition that exists only in a Markdown file will drift from
the number it describes; one that the receipt emits cannot.

**Failures stay in the denominator.** No reducer may silently drop a session that
lacks a measure. Percentiles are conditional on outcomes that possess the measure,
and the population count travels with them.

## Adding a transport adapter

An adapter owns protocol mechanics and timestamps — nothing else. It takes a
`SessionRequest`, returns a `SessionOutcome`, and never learns what workload it is
part of or how the result will be reduced.

1. Add the client library as its own extra in `pyproject.toml`.
2. Register the extra in `_TRANSPORT_EXTRAS` in `tests/test_core_boundary.py`, which
   both permits the import and confines it to the adapter subtree.
3. Implement the `TransportAdapter` protocol under `src/asrbench/adapters/`.
4. Add a test that drives it without the network, so the interface is exercised on
   the default tier.

Diagnostics are bounded lifecycle edges, never per-chunk logging.

## Changing the receipt schema

The receipt is the unit of truth, and published results cite it. Before 1.0 the schema
may change without a migration path, but never silently: bump the schema version
string in the same change, and say in the commit message what a consumer of the old
version must do.

## Corpora

Commit recipes and digests, never audio. Every entry in `datasets.lock` carries its
licence and the attribution a published result must reproduce. Derived audio —
excerpts, decodes, concatenations, constructed fixtures — inherits the source
corpus's licence.

Do not add a corpus that requires registration or carries a non-commercial or
no-derivatives restriction. A benchmark others cannot legally run is not a benchmark.

## Commits and licensing

Sign off every commit with `git commit -s`, certifying the
[Developer Certificate of Origin](https://developercertificate.org/).

Contributions are accepted under the Apache License, Version 2.0 — the licence this
project is distributed under, per its §5. Do not contribute material you cannot
license that way, and do not introduce material from a non-public source. Sources of
record here are public and reachable: published papers, public model cards, public
vendor documentation, and public client repositories.
