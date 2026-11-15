"""A throwaway private CA, server, and client certificate, via openssl.

Test-only key material, generated per test into a temporary directory.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SERVER_NAME = "nim.site.internal"


@dataclass(frozen=True)
class SiteTls:
    """Paths to PEM files for one private CA."""

    ca: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def _openssl(*args: str) -> None:
    subprocess.run(
        ["openssl", *args], check=True, capture_output=True, text=True
    )


def available() -> bool:
    """Return whether an openssl binary is on PATH."""
    return shutil.which("openssl") is not None


def make_site_tls(directory: Path) -> SiteTls:
    """Write a CA, a server cert for :data:`SERVER_NAME`, and a client cert.

    Args:
        directory: Where to write the PEM files.

    Returns:
        The file paths.
    """
    ca_key = directory / "ca.key"
    ca = directory / "ca.pem"
    _openssl(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=asrbench test CA", "-keyout", str(ca_key),
        "-out", str(ca),
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
    )  # fmt: skip
    pairs: dict[str, tuple[Path, Path]] = {}
    for name, subject, extension in (
        ("server", SERVER_NAME, f"subjectAltName=DNS:{SERVER_NAME}"),
        ("client", "asrbench-client", "extendedKeyUsage=clientAuth"),
    ):
        key = directory / f"{name}.key"
        csr = directory / f"{name}.csr"
        cert = directory / f"{name}.pem"
        ext = directory / f"{name}.ext"
        ext.write_text(extension + "\n", encoding="utf-8")
        _openssl(
            "req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={subject}",
            "-keyout", str(key), "-out", str(csr),
        )  # fmt: skip
        _openssl(
            "x509", "-req", "-in", str(csr), "-CA", str(ca),
            "-CAkey", str(ca_key), "-CAcreateserial", "-days", "1",
            "-extfile", str(ext), "-out", str(cert),
        )  # fmt: skip
        pairs[name] = (cert, key)
    return SiteTls(
        ca=ca,
        server_cert=pairs["server"][0],
        server_key=pairs["server"][1],
        client_cert=pairs["client"][0],
        client_key=pairs["client"][1],
    )
