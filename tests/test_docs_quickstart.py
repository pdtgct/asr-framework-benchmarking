"""The README quickstart runs against the mock adapter.

The quickstart shells out to ``uv run``, which syncs the project and may
need to resolve packages from an index. On an air-gapped host, or in a
network namespace with no network, that resolution fails before any
asrbench code runs. That is an environment limit, not a regression, so
the test is skipped with the reason. Any other ``uv run`` failure still
fails the test.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
#: Text uv prints when it could not resolve from a package index. This is
#: broader than "unreachable": ``Failed to fetch`` also covers a private
#: index answering 401/403/5xx. Matched only on a failed probe, and every
#: such failure happens before any asrbench code runs, so a working online
#: sync is never skipped.
_OFFLINE = re.compile(
    r"network was disabled|Failed to fetch|dns error|"
    r"failed to lookup address|Temporary failure in name resolution|"
    r"error sending request for url|client error \(Connect\)",
    re.IGNORECASE,
)


def offline_skip_reason(returncode: int, stderr: str) -> str | None:
    """Return a skip reason when a failed ``uv run`` could not resolve.

    Args:
        returncode: Exit code of the probe.
        stderr: Its standard error.

    Returns:
        A reason, or ``None`` when the probe passed or failed for some
        other reason.
    """
    if returncode == 0 or not _OFFLINE.search(stderr):
        return None
    tail = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    return f"uv cannot resolve packages offline: {tail}"


def _require_uv(env: dict[str, str]) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    probe = subprocess.run(
        'uv run python -c "import asrbench"',
        shell=True,
        cwd=_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    reason = offline_skip_reason(probe.returncode, probe.stderr)
    if reason is not None:
        pytest.skip(reason)


def test_offline_skip_reason_only_matches_resolution_failures() -> None:
    assert offline_skip_reason(0, "Failed to fetch") is None
    assert offline_skip_reason(1, "SyntaxError: bad") is None
    reason = offline_skip_reason(
        2, "cause: Failed to fetch: https://pypi.org/simple/hatchling/\n"
    )
    assert reason is not None and "offline" in reason
    hint = "hint: Packages were unavailable because the network was disabled."
    assert offline_skip_reason(1, hint) is not None
    dns = "error: dns error: failed to lookup address information"
    assert offline_skip_reason(1, dns) is not None
    assert offline_skip_reason(1, "") is None


def test_readme_quickstart_block_runs() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "<!-- asrbench:doc-test -->"
    assert marker in readme
    section = readme.split(marker, 1)[1]
    script = section.split("```bash\n", 1)[1].split("```", 1)[0]
    commands = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert commands
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ASRBENCH_")
    }
    _require_uv(env)
    produced = [_ROOT / "receipt.json", _ROOT / "receipt-again.json"]
    try:
        for command in commands:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
        first = json.loads(produced[0].read_text(encoding="utf-8"))
        second = json.loads(produced[1].read_text(encoding="utf-8"))
        assert first["comparison_key"] == second["comparison_key"]
        left = first["resolved_config"]
        right = second["resolved_config"]
        left["run"]["output_path"] = None
        right["run"]["output_path"] = None
        assert left == right
    finally:
        for path in produced:
            path.unlink(missing_ok=True)
