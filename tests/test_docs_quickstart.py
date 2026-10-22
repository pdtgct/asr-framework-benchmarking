"""The README quickstart runs against the mock adapter."""

import json
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).parents[1]


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
