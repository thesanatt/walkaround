"""Fast one_bit mask decoding for the keyframe pass (panels lane, 11/1).

meta_sam_parser decodes a one_bit mask with a pure-Python context-adaptive range decoder and then encodes the raster
again to prove the payload canonical, about 1.6 us per mask pixel; on the 20 s pair C walks that was 78 s of CPU per
walk, four fifths of it on the "car" masks the keyframe containment test reads (docs/BUILD_STATUS.md section 4 item 9).

decode_one_bit is a port of the same decoder that returns the same raster bit for bit (tests/test_fastmask.py checks
it against the library on the stored pair C lanes and the conformance cases). Three things make it faster: the context
bits of the two rows above are built per row with numpy, the range coder is inlined, and the library's canonical
re-encode is replaced by a check that the payload is used up exactly (a payload that is cut short, or carries bytes
past its end, still raises). rows= stops after that many rows and leaves the rest false, which is all a containment
test on the upper part of a mask needs. Lossless payloads, and any payload this path refuses, go to the library
(sam_wire.decode_mask), so the library stays the decoder of record and the oracle.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from meta_sam_parser import InvalidSegmentationMaskError

from . import sam_wire
from .contracts import MaskEncoding, MaskRef

try:  # the library's base-85 unpacking, with its own canonical-form check
    from meta_sam_parser._mask_codec import _unpack as _unpack85
except ImportError:  # a later parser without the private helper: every mask goes through the library
    _unpack85 = None

_M32 = 0xFFFFFFFF
_TOP = 1 << 24
_BOTTOM = 1 << 16
_HIGH = _M32 - _TOP  # a range above this can wrap past 2^32 and keep its top byte, so it takes the full test
_CONTEXTS = 1 << 12
_INCREMENT = 14
_COUNT_LIMIT = 4096


def available() -> bool:
    """True when the fast path can run (the parser still has the helper it borrows)."""
    return _unpack85 is not None


def decode_one_bit(payload: str, width: int, height: int, *, rows: int | None = None) -> np.ndarray:
    """Bool (height, width) raster of a one_bit payload, the same as meta_sam_parser.decode_mask_to_raster. With rows,
    only the first rows rows are decoded and the rest stay false. InvalidSegmentationMaskError when the payload is
    malformed, cut short or longer than the raster needs."""
    if _unpack85 is None:
        raise InvalidSegmentationMaskError("the fast decoder is not available with this meta_sam_parser")
    if not isinstance(payload, str) or not payload.startswith("!"):
        raise InvalidSegmentationMaskError("one_bit mask payloads must start with !.")
    if width <= 0 or height <= 0:
        raise InvalidSegmentationMaskError("Mask dimensions must be positive safe integers.")
    data = _unpack85(payload[1:])
    n = len(data)
    if n < 4:
        raise InvalidSegmentationMaskError("Mask payload is missing its finalization.")
    d = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]  # code minus low, mod 2^32
    pos = 4
    low = 0
    rng = _M32
    c0 = [1] * _CONTEXTS
    c1 = [1] * _CONTEXTS
    w = int(width)
    out = np.zeros((int(height), w), dtype=np.uint8)
    up1 = np.zeros(w + 4, dtype=np.int32)  # the row above, two zero columns each side
    up2 = np.zeros(w + 4, dtype=np.int32)  # the row above that
    last = int(height) if rows is None else max(0, min(int(height), int(rows)))
    try:
        for y in range(last):
            above = (
                (up1[0:w] << 2) | (up1[1 : w + 1] << 3) | (up1[2 : w + 2] << 4) | (up1[3 : w + 3] << 5) | (up1[4 : w + 4] << 6)
                | (up2[0:w] << 7) | (up2[1 : w + 1] << 8) | (up2[2 : w + 2] << 9) | (up2[3 : w + 3] << 10) | (up2[4 : w + 4] << 11)
            ).tolist()
            bits: list[int] = []
            put = bits.append
            left = 0  # bit 0: the pixel to the left, bit 1: the one before it
            for ctx in above:
                ctx |= left
                zero = c0[ctx]
                one = c1[ctx]
                scaled = rng // (zero + one)
                t = scaled * zero
                if d < t:
                    rng = t
                    zero += _INCREMENT
                    c0[ctx] = zero
                    left = (left << 1) & 2
                    put(0)
                else:
                    low = (low + t) & _M32
                    d -= t
                    rng = scaled * one
                    one += _INCREMENT
                    c1[ctx] = one
                    left = ((left << 1) & 2) | 1
                    put(1)
                if zero + one >= _COUNT_LIMIT:
                    c0[ctx] = (zero >> 1) or 1
                    c1[ctx] = (one >> 1) or 1
                if not _TOP <= rng <= _HIGH:
                    while True:
                        if (low ^ ((low + rng) & _M32)) < _TOP:
                            pass
                        elif rng < _BOTTOM:
                            rng = (-low) & (_BOTTOM - 1)
                        else:
                            break
                        d = ((d << 8) | data[pos]) & _M32
                        pos += 1
                        low = (low << 8) & _M32
                        rng = (rng << 8) & _M32
            row = np.array(bits, dtype=np.uint8)
            out[y] = row
            up1, up2 = up2, up1
            up1[2 : w + 2] = row
    except IndexError as exc:
        raise InvalidSegmentationMaskError("Mask payload ended before decoding completed.") from exc
    except ZeroDivisionError as exc:
        raise InvalidSegmentationMaskError("Mask payload has invalid coding state.") from exc
    if rows is None and pos != n:
        raise InvalidSegmentationMaskError(f"Mask payload carries {n - pos} bytes past the end of its raster.")
    return out.astype(bool)


def decode_ref(ref: MaskRef, payload: str, *, rows: int | None = None) -> np.ndarray:
    """One mask of a lane: one_bit through decode_one_bit, lossless (and anything it refuses) through the library."""
    if ref.encoding is MaskEncoding.ONE_BIT and _unpack85 is not None:
        try:
            return decode_one_bit(payload, ref.w, ref.h, rows=rows)
        except InvalidSegmentationMaskError:
            pass  # the library gives the error of record (or, for a payload only this path refuses, the raster)
    return sam_wire.decode_mask(ref, payload)


class LaneMasks:
    """Decodes masks of stored lanes in the calling thread, reading each lane's text once. stats() is (masks decoded,
    CPU seconds of this thread spent decoding), the same shape as sam_wire.DecodePool.stats()."""

    def __init__(self) -> None:
        self._texts: dict[Path, str] = {}
        self._decoded = 0
        self._cpu_s = 0.0

    def text(self, lane_path: Path) -> str:
        path = Path(lane_path)
        if path not in self._texts:
            self._texts[path] = sam_wire.lane_text(sam_wire.read_events(path))
        return self._texts[path]

    def decode(self, lane_path: Path, refs: list[MaskRef], *, rows: int | None = None) -> list[np.ndarray]:
        """Rasters for refs (rows as in decode_one_bit); a bad payload raises InvalidSegmentationMaskError."""
        if not refs:
            return []
        text = self.text(lane_path)
        start = time.thread_time()
        try:
            out = [decode_ref(ref, sam_wire.mask_payload(text, ref), rows=rows) for ref in refs]
        finally:
            self._cpu_s += time.thread_time() - start
        self._decoded += len(out)
        return out

    def stats(self) -> tuple[int, float]:
        return self._decoded, self._cpu_s


__all__ = ["LaneMasks", "available", "decode_one_bit", "decode_ref"]
