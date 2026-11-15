"""Narrowband audio conditions, in the standard library.

A telephony comparison sends the same clips band-limited to 8 kHz. Two
ways to offer that, and a server may accept only one:

- ``telephony_8k``: decimate by two and send 8 kHz PCM16, declaring
  ``sample_rate_hertz = 8000``. What a telephone path delivers.
- ``telephony_8k_in_16k``: decimate by two, then interpolate back to
  16 kHz and send at 16 kHz. The same narrowband content, for a server or
  model that only accepts 16 kHz.

The filter is fixed and versioned (:data:`RESAMPLER`) so two receipts
that name it offered byte-identical audio. It is a 127-tap
Blackman-windowed sinc low-pass at 3.8 kHz, unity DC gain, applied as a
polyphase decimator and interpolator. Output is rounded half away from
zero and clipped to int16. No numpy, so the core stays stdlib-only;
throughput is about 10x real time on one core, done once per clip at
materialization.
"""

from __future__ import annotations

import array
import math
import sys

from asrbench.audio import SAMPLE_RATE

NARROWBAND_RATE = 8_000
#: Name recorded on receipts. Changing any constant below is a new name.
RESAMPLER = "fir_blackman_sinc_127tap_fc3800_v1"
_TAPS = 127
_CUTOFF_HZ = 3_800.0


def _design() -> tuple[float, ...]:
    centre = (_TAPS - 1) / 2
    fc = _CUTOFF_HZ / SAMPLE_RATE
    taps: list[float] = []
    for n in range(_TAPS):
        x = n - centre
        sinc = (
            2 * fc if x == 0 else math.sin(2 * math.pi * fc * x) / (math.pi * x)
        )
        window = (
            0.42
            - 0.5 * math.cos(2 * math.pi * n / (_TAPS - 1))
            + 0.08 * math.cos(4 * math.pi * n / (_TAPS - 1))
        )
        taps.append(sinc * window)
    total = sum(taps)
    return tuple(tap / total for tap in taps)


_FILTER = _design()


def _samples(pcm: bytes) -> list[int]:
    values = array.array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _pcm(values: list[float]) -> bytes:
    out = array.array("h")
    for value in values:
        rounded = int(math.copysign(math.floor(abs(value) + 0.5), value))
        out.append(max(-32768, min(32767, rounded)))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def decimate_16k_to_8k(pcm: bytes) -> bytes:
    """Low-pass at 3.8 kHz and keep every second sample.

    Args:
        pcm: Mono PCM16 at 16 kHz.

    Returns:
        Mono PCM16 at 8 kHz, ``len(pcm) // 4`` samples rounded up.
    """
    source = _samples(pcm)
    half = _TAPS // 2
    padded = [0] * half + source + [0] * half
    taps = _FILTER
    out = [
        math.fsum(map(float.__mul__, taps, map(float, padded[i : i + _TAPS])))
        for i in range(0, len(source), 2)
    ]
    return _pcm(out)


def interpolate_8k_to_16k(pcm: bytes) -> bytes:
    """Zero-stuff by two and low-pass at 3.8 kHz with gain two.

    Args:
        pcm: Mono PCM16 at 8 kHz.

    Returns:
        Mono PCM16 at 16 kHz, twice as many samples.
    """
    source = [float(value) for value in _samples(pcm)]
    half = _TAPS // 2
    banks = (_FILTER[0::2][::-1], _FILTER[1::2][::-1])
    pad = len(banks[0])
    padded = [0.0] * pad + source + [0.0] * pad
    out: list[float] = []
    for m in range(2 * len(source)):
        # y[m] = 2 * sum_n h[n] * up[m + half - n], and up[j] is x[j / 2]
        # at even j, zero at odd j, so only taps of one parity contribute.
        parity = (m + half) % 2
        bank = banks[parity]
        centre = (m + half - parity) // 2
        window = padded[centre - len(bank) + 1 + pad : centre + 1 + pad]
        out.append(2.0 * math.fsum(map(float.__mul__, bank, window)))
    return _pcm(out)
