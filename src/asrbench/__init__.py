"""A protocol-level benchmark for streaming speech recognition servers.

The measurement core is deliberately stdlib-only. A transport adapter may
depend on a client library, declared as an extra; the runner, the schemas,
the reducers, and the reporters may not. That constraint is what lets the
framework run with no GPU, no server, and no network, which is the
difference between a benchmark others can reproduce and one they can only
read about.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
