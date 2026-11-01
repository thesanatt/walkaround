"""Offline tests for walkaround.fastmask: the fast one_bit decoder gives meta_sam_parser's raster bit for bit (the 10/26
fixtures, the conformance corpus, rasters the library encodes, a sample of the stored pair C lanes when present), stops
early on rows=, refuses cut or padded payloads, and leaves lossless payloads to the library. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from meta_sam_parser import InvalidSegmentationMaskError
from meta_sam_parser._mask_codec import _encode_raster, _pack, _unpack

from walkaround import contracts as c
from walkaround import fastmask, sam_wire
from walkaround.contracts import MaskEncoding, MaskRef

ROOT = Path(__file__).resolve().parent
FIXTURES = [ROOT / "fixtures" / "sam" / "pingpong_text.events.ndjson", ROOT / "fixtures" / "sam" / "pingpong_box.events.ndjson"]
CASES = sorted((ROOT / "sam_conformance" / "cases").glob("*.json"))
PAIR_C = ROOT.parent / "runs" / "pairC_core1" / "sam"


def _both(ref: MaskRef, payload: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        lib = sam_wire.decode_mask(ref, payload)
    except InvalidSegmentationMaskError:
        lib = None
    try:
        fast = fastmask.decode_ref(ref, payload)
    except InvalidSegmentationMaskError:
        fast = None
    return lib, fast


def _ref(raster: np.ndarray) -> tuple[MaskRef, str]:
    h, w = raster.shape
    payload = _encode_raster(raster.astype(np.uint8).ravel().tolist(), w, h)
    return MaskRef(h=h, w=w, encoding=MaskEncoding.ONE_BIT, payload_offset=0, payload_len=len(payload)), payload


def test_the_fast_path_is_available() -> None:
    assert fastmask.available()


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_every_fixture_mask_decodes_bit_for_bit(path: Path) -> None:
    events = list(sam_wire.read_events(path))
    frames, _ = sam_wire.index_from_events(events, media="video")
    text = sam_wire.lane_text(events)
    refs = [o.mask for fr in frames for o in fr.objects if o.mask is not None]
    assert len(refs) == 134
    for ref in refs:
        payload = sam_wire.mask_payload(text, ref)
        assert np.array_equal(fastmask.decode_one_bit(payload, ref.w, ref.h), sam_wire.decode_mask(ref, payload))


def _case_events(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The text deltas and done events of a conformance case, enough to rebuild its lane text and mask refs."""
    out: list[dict[str, Any]] = []
    for e in case["events"]:
        base = {k: v for k, v in (e.get("lane") or {}).items() if k in ("item_id", "output_index", "content_index")}
        if e["type"] == "output_text_delta":
            out.append({"type": "response.output_text.delta", **base, "delta": case["chunks"][e["chunk"]]})
        elif e["type"] == "output_text_done":
            out.append({"type": "response.output_text.done", **base, "text": e["text"]})
    out.append({"type": "response.completed", "response": {"id": "resp_case", "status": "completed"}})
    return out


def test_the_conformance_masks_agree_with_the_library_valid_or_not() -> None:
    compared = 0
    for path in CASES:
        case = json.loads(path.read_text())
        if case.get("source", {}).get("operation") in ("iterator", "next"):
            continue
        events = _case_events(case)
        frames, _ = sam_wire.index_from_events(events, media=case["media"])
        text = sam_wire.lane_text(events)
        for fr in frames:
            for obj in fr.objects:
                if obj.mask is None:
                    continue
                lib, fast = _both(obj.mask, sam_wire.mask_payload(text, obj.mask))
                assert (lib is None) == (fast is None), path.stem
                if lib is not None:
                    assert np.array_equal(lib, fast), path.stem
                compared += 1
    assert compared >= 10


def test_rasters_the_library_encodes_come_back_unchanged() -> None:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:120, 0:200]
    shapes = [
        np.zeros((5, 7), dtype=bool),
        np.ones((9, 4), dtype=bool),
        np.eye(30, dtype=bool),
        ((xx - 90) ** 2 / 80.0**2 + (yy - 60) ** 2 / 45.0**2) <= 1.0,  # an ellipse, like a wheel
        rng.random((40, 60)) < 0.5,  # noise: every context and the count halving
        np.ones((1, 300), dtype=bool),
        np.zeros((300, 1), dtype=bool),
    ]
    for raster in shapes:
        ref, payload = _ref(raster)
        out = fastmask.decode_one_bit(payload, ref.w, ref.h)
        assert out.dtype == bool and out.shape == raster.shape and np.array_equal(out, raster)


def test_rows_decodes_the_top_of_the_mask_and_leaves_the_rest_false() -> None:
    yy, xx = np.mgrid[0:80, 0:100]
    raster = ((xx - 50) ** 2 + (yy - 40) ** 2) <= 35**2
    ref, payload = _ref(raster)
    for rows in (0, 1, 17, 79, 80, 500):
        out = fastmask.decode_one_bit(payload, ref.w, ref.h, rows=rows)
        k = min(rows, 80)
        assert np.array_equal(out[:k], raster[:k]) and not out[k:].any()


def test_a_cut_padded_or_mislabelled_payload_raises() -> None:
    raster = np.zeros((20, 30), dtype=bool)
    raster[5:15, 8:22] = True
    ref, payload = _ref(raster)
    data = _unpack(payload[1:])
    with pytest.raises(InvalidSegmentationMaskError):
        fastmask.decode_one_bit("!" + _pack(data + b"\x00\x00"), ref.w, ref.h)  # bytes past the raster's end
    with pytest.raises(InvalidSegmentationMaskError):
        fastmask.decode_one_bit("!" + _pack(data[:3]), ref.w, ref.h)  # no finalization
    with pytest.raises(InvalidSegmentationMaskError):
        fastmask.decode_one_bit("~" + payload[1:], ref.w, ref.h)
    with pytest.raises(InvalidSegmentationMaskError):
        fastmask.decode_one_bit(payload[:-3], ref.w, ref.h)  # not canonical base 85 any more
    big = MaskRef(h=200, w=300, encoding=MaskEncoding.ONE_BIT, payload_offset=0, payload_len=len(payload))
    lib, fast = _both(big, payload)  # the payload of a smaller raster runs out early in both decoders
    assert lib is None and fast is None


def test_lossless_payloads_go_to_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    case = json.loads((ROOT / "sam_conformance" / "cases" / "captured-api-image-lossless.json").read_text())
    events = _case_events(case)
    frames, _ = sam_wire.index_from_events(events, media="image")
    text = sam_wire.lane_text(events)
    ref = frames[0].objects[0].mask
    assert ref is not None and ref.encoding is MaskEncoding.LOSSLESS

    def refuse(*args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("a lossless payload reached the one_bit decoder")

    monkeypatch.setattr(fastmask, "decode_one_bit", refuse)
    assert np.array_equal(fastmask.decode_ref(ref, sam_wire.mask_payload(text, ref)), sam_wire.decode_mask(ref, sam_wire.mask_payload(text, ref)))


def test_lane_masks_reads_each_lane_once_and_counts_its_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[Path] = []
    real = sam_wire.read_events

    def counting(path: Path):
        reads.append(Path(path))
        return real(path)

    monkeypatch.setattr(sam_wire, "read_events", counting)
    frames, _ = sam_wire.index_from_events(real(FIXTURES[0]), media="video")
    refs = [o.mask for fr in frames for o in fr.objects if o.mask is not None]
    masks = fastmask.LaneMasks()
    first = masks.decode(FIXTURES[0], refs[:3])
    second = masks.decode(FIXTURES[0], refs[3:5], rows=2)
    assert reads == [FIXTURES[0]] and len(first) == 3 and len(second) == 2
    assert not second[0][2:].any()
    decoded, cpu_s = masks.stats()
    assert decoded == 5 and cpu_s >= 0.0


@pytest.mark.skipif(not PAIR_C.is_dir(), reason="runs/pairC_core1 is not on this machine (runs/ is not in the repo)")
def test_a_sample_of_the_pair_c_lanes_decodes_bit_for_bit() -> None:
    """Every 150th mask of the 10/31 live run's walk lanes. The full sweep (10,547 masks, 455 million pixels, 0
    differences) is in docs/CONTRACTS.md under the panels lane notes."""
    checked = 0
    for gz in sorted(PAIR_C.glob("*/*.ndjson.gz")):
        if gz.name.startswith("sp_"):
            continue
        index = gz.with_name(gz.name.replace(".ndjson.gz", ".index.json"))
        frames = c.read_json_list(index, c.IndexFrame) if index.is_file() else []
        refs = [o.mask for fr in frames for o in fr.objects if o.mask is not None][::150]
        if not refs:
            continue
        text = sam_wire.lane_text(sam_wire.read_events(gz))
        for ref in refs:
            payload = sam_wire.mask_payload(text, ref)
            assert np.array_equal(fastmask.decode_one_bit(payload, ref.w, ref.h), sam_wire.decode_mask(ref, payload)), gz.name
            checked += 1
    assert checked >= 40
