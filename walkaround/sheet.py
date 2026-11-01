"""The evidence sheet (docs/ARCHITECTURE.md section 7 item 6; docs/CONTRACTS.md sheet M13).

build_sheet reads the run's files and fills contracts.SheetModel; render_html turns it into plain HTML with Jinja2;
render_pdf prints it with Playwright Chromium when that is installed. Nothing here recomputes a verdict or calls a
model. The only pixels this module writes are the blurred crops with the marks drawn on them (sheet/frames/<kf>.jpg)
and, for a keep_plate run, the run owner's variant with the rented car's plate kept (sheet/frames/<kf>.owner.jpg),
made from the quarantined original while it exists and left alone once it is gone.

sheet.json must rebuild byte for byte after the run ends, so it reads only state that is frozen when it is first
written: the event log up to run.json's events_head_hash (frozen right before the sheet stage), deletion rows already
announced on that chain or timestamped before its head, and checks that do not depend on the quarantine still being
there. bundle/manifest_run.json is written after the sheet and is never read here.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from . import ledger, paths, privacy, rules, seal
from .contracts import (
    BODY_PANELS,
    LOWER_BUMPERS,
    WHEEL_PANELS,
    Calibration,
    Claim,
    Contract,
    Coverage,
    Crop,
    DeletionClocks,
    DeletionKind,
    DeletionRecord,
    Deletions,
    Event,
    EventKind,
    Keyframe,
    Mark,
    MarkKind,
    MarkTrack,
    Narration,
    Noun,
    Panel,
    PanelId,
    PanelMatch,
    PanelTrack,
    ResponsesStore,
    RulerKind,
    RunManifest,
    SealManifest,
    SheetChip,
    SheetFrame,
    SheetMethod,
    SheetModel,
    SheetNotComparable,
    SheetOriginal,
    SheetPanelBlock,
    SheetVerdictLine,
    SizeEstimate,
    Stamps,
    StaysPutVerdict,
    ThresholdType,
    Verdict,
    VerdictLabel,
    VideoRecord,
    read_json,
    read_json_list,
    write_json,
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
HERTZ_REV = "REV 7.2025"
HERTZ_SAVED = "2026-10-26"
HERTZ_NOTE = (
    f"Example thresholds Hertz published ({HERTZ_REV}), illustrative only, quoted from its Damage Charges support page "
    f"saved {HERTZ_SAVED}: a body scratch is billable when it cannot be covered by a credit card (2 in x 3.5 in), or when "
    "scratches over 1 in together exceed that area; a dent with a diameter over 1 in, or multiple dents; a wheel, rim or "
    "hubcap scratch or scuff over 1 in; any crack or chip in glass. Paint transfer or scuffs that do not break the paint "
    "are listed as Ordinary Wear and not charged."
)
ORDINARY_WEAR_LINE = f"Hertz's page lists scuffs that do not break paint as ordinary wear ({HERTZ_REV}, illustrative only; page saved {HERTZ_SAVED})."
NO_RULER_LINE = "size not stated: no ruler within 0.6 m of this mark"
WHAT_SEAL_DOES_NOT_PROVE = [
    "which car was filmed, unless the plate was kept on this sheet at the renter's request",
    "that the walk was not staged before the upload; the seal dates the bytes, not the events in them",
    "anything about the pickup condition when the seal was made after the return",
]
LABEL_TEXT = {
    VerdictLabel.ALREADY_THERE.value: "already there",
    VerdictLabel.NEEDS_YOUR_EYE.value: "needs your eye",
    VerdictLabel.NEW_OR_UNSEEN.value: "new or unseen",
    VerdictLabel.NOT_SEEN_AT_RETURN.value: "not seen at return",
    VerdictLabel.NOT_COMPARABLE.value: "not comparable",
}
_LABEL_BGR = {
    VerdictLabel.ALREADY_THERE: (79, 106, 45),
    VerdictLabel.NEEDS_YOUR_EYE: (0, 103, 154),
    VerdictLabel.NEW_OR_UNSEEN: (38, 34, 155),
    VerdictLabel.NOT_SEEN_AT_RETURN: (140, 90, 58),
    VerdictLabel.NOT_COMPARABLE: (102, 102, 102),
}
_TIRE_CODE = re.compile(r"^\s*(\d{3})\s*/\s*(\d{2})\s*[A-Za-z]*\s*(\d{2}(?:\.\d)?)\s*$")
RUN_END_NOTE = (
    "Steps marked not yet happen at run end, after this sheet was written; deletions.json and "
    "bundle/manifest_run.json record them."
)
_TSA_VERIFY = {
    "freetsa": (
        "freetsa's certificates (download https://freetsa.org/files/cacert.pem and https://freetsa.org/files/tsa.crt)",
        "openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile cacert.pem -untrusted tsa.crt",
    ),
    # TODO(D1 E12): confirm a DigiCert token verifies against the system bundle alone (no -untrusted intermediate).
    "DigiCert": (
        "DigiCert's root from your system's CA bundle (the token carries DigiCert's signing certificate)",
        "openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile /etc/ssl/cert.pem",
    ),
}
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


# Small helpers -------------------------------------------------------------------------------------------------------


def display_name(panel_id: PanelId) -> str:
    return panel_id.value.replace("_", " ")


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _read_opt(path: Path, model: type[Contract]) -> Any:
    return read_json(path, model) if Path(path).is_file() else None


def _read_list_opt(path: Path, model: type[Contract]) -> list[Any]:
    return read_json_list(path, model) if Path(path).is_file() else []


def _side(role: str) -> str | None:
    """pickup and top-ups are the pickup side; return is the return side; claim roles are neither."""
    if role == "pickup" or role.startswith("topup_"):
        return "pickup"
    return "return" if role == "return" else None


def _role_order(role: str) -> tuple[int, str]:
    rank = {"pickup": 0, "return": 2}.get(role, 1 if role.startswith("topup_") else 3)
    return rank, role


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _wall_clock(camera_time: str, pts: float) -> str | None:
    if seal.normalize_time(camera_time) == seal.NOT_PRESENT:
        return None
    return seal._iso_utc(seal._parse_iso(camera_time) + timedelta(seconds=float(pts)))


def _block_of(ots_status: str | None) -> int | None:
    if ots_status and ots_status.startswith("confirmed:"):
        try:
            return int(ots_status.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _best_ots(*statuses: str | None) -> str | None:
    """The strongest OTS status: confirmed (lowest block) over pending over failed over none. The manifest's proof
    counts for each original because manifest.json holds the original's sha256."""
    confirmed = [s for s in statuses if _block_of(s) is not None]
    if confirmed:
        return min(confirmed, key=lambda s: _block_of(s) or 0)
    rank = {"pending": 2, "failed": 1}
    present = [s for s in statuses if s]
    return max(present, key=lambda s: rank.get(s.split(":", 1)[0], 0), default=None)


def ots_note(status: str | None) -> str:
    """Page 1's words for one OpenTimestamps status, with a short reason instead of the raw error."""
    block = _block_of(status)
    if block is not None:
        return f"Bitcoin block {block}"
    if not status:
        return "OpenTimestamps not stamped"
    if status.startswith("pending"):
        return "OpenTimestamps pending at the calendars (ots upgrade adds the Bitcoin block after a few hours)"
    if status.startswith("failed:"):
        reason = re.sub(r"^[A-Za-z]+(Error|Exception|Expired): ", "", status.removeprefix("failed:"))
        if "not installed" in reason:
            reason = "the OpenTimestamps client is not installed on the server"
        return f"OpenTimestamps not stamped ({reason[:120]})"
    return f"OpenTimestamps {status}"


def _before(ts: str, limit: datetime) -> bool:
    try:
        return seal._parse_iso(ts) < limit
    except ValueError:
        return False


def tire_diameter_in(code: str | None) -> float | None:
    """D = 2 x (SW x AR / 2540 + RD / 2) in inches from a sidewall code like 225/60R16; None for an unknown code."""
    m = _TIRE_CODE.match(code or "")
    if not m:
        return None
    sw, ar, rd = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return 2 * (sw * ar / 2540 + rd / 2)


# The sentences of section 7 ------------------------------------------------------------------------------------------


def size_line(size: SizeEstimate | None) -> str:
    """'about 0.9 in, between 0.6 and 1.3 in (ruler: the license plate, 12 in)' or 'size not stated: <reason>', each
    followed by why every other ruler next to the mark did not size it (SizeEstimate.ruler_notes)."""
    if size is None:
        return NO_RULER_LINE
    notes = "".join(f"; {note}" for note in size.ruler_notes)
    if size.refused_reason:
        return f"size not stated: {size.refused_reason}{notes}"
    if size.inches is None:
        return "size not stated: no ruler qualified"
    if size.ruler == RulerKind.PLATE:
        ruler = "the license plate, 12 in"
    elif size.ruler == RulerKind.WHEEL:
        diameter = tire_diameter_in(size.tire_code)
        ruler = f"the wheel, {diameter:.1f} in from {size.tire_code.strip()}, typed by you" if diameter else "the wheel"
    else:
        ruler = "not recorded"
    text = f"about {size.inches:.1f} in"
    if size.bound_lo is not None and size.bound_hi is not None:
        text += f", between {size.bound_lo:.1f} and {size.bound_hi:.1f} in"
    return f"{text} (ruler: {ruler}{notes})"


def threshold_line(size: SizeEstimate | None, th: dict[str, Any]) -> str | None:
    """The Hertz comparison for a sized mark, refused (not compared) when the bound straddles the threshold."""
    if size is None or size.inches is None or size.threshold_type in (None, ThresholdType.BODY_SCUFF):
        return None
    table = {
        ThresholdType.BODY_SCRATCH: (th["ruler"]["threshold_body_scratch_in"], "a body scratch along its long axis"),
        ThresholdType.DENT: (th["ruler"]["threshold_dent_in"], "a dent"),
        ThresholdType.WHEEL_MARK: (th["ruler"]["threshold_wheel_in"], "a wheel scratch or scuff"),
    }
    limit, name = table[size.threshold_type]
    lo = size.bound_lo if size.bound_lo is not None else size.inches
    hi = size.bound_hi if size.bound_hi is not None else size.inches
    tail = f"Hertz's example threshold of {limit:g} in for {name} ({HERTZ_REV}, illustrative only)"
    if hi < limit:
        return f"under {tail}"
    if lo > limit:
        return f"over {tail}"
    return f"threshold not compared: the size bound straddles {limit:g} in, {tail}"


def ordinary_wear_line(mark_type: MarkKind | None, panel_id: PanelId) -> str | None:
    """The Hertz sentence for a scuff on a body panel, else None."""
    if mark_type == MarkKind.SCUFF and panel_id in BODY_PANELS:
        return ORDINARY_WEAR_LINE
    return None


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


STAYSPUT_UNCALIBRATED = "recorded only: Stays Put is not calibrated yet and changes no label"


def neighbour_line(mt: MarkTrack | None) -> str | None:
    """Stays Put by neighbours in words (docs/STAYS_PUT_ANCHORS.md; web/src/lib/neighbours.ts says the same): 'neighbour
    test: moves with its 6 neighbours, 0.6x their wobble over the clip and 0.8x within 0.5 s; fixed'. None when the
    clip has no neighbour test or its lane gave no answer."""
    nt = mt.neighbour_test if mt is not None else None
    if nt is None or nt.verdict == StaysPutVerdict.NOT_RUN:
        return None
    if nt.verdict == StaysPutVerdict.NO_TRACK:
        return "neighbour test: SAM did not follow the mark through enough of the clip"
    if nt.ratio_full is None or nt.ratio_near is None:
        return f"neighbour test: {nt.reason or 'too few frames to measure'}; inconclusive"
    ratios = f"{nt.ratio_full:.1f}x {{}} over the clip and {nt.ratio_near:.1f}x within {nt.near_s:g} s"
    if nt.verdict == StaysPutVerdict.FIXED:
        words = f"moves with its {nt.used} neighbours, " + ratios.format("their wobble")
    elif nt.verdict == StaysPutVerdict.SLIDING:
        words = f"slides against its {nt.used} neighbours, " + ratios.format("their wobble")
    else:
        words = ratios.format(f"its {nt.used} neighbours' wobble")
    return f"neighbour test: {words}; {nt.verdict.value}"


def staysput_line(mt: MarkTrack | None, *, calibrated: bool = True) -> str | None:
    """'drift 0.03, b1 0.20, b2 0.01, not panned; fixed'; an inconclusive track says which test it stopped at: the
    camera's motion (section 14 item 18), too few whitened frames, or a drift between fixed and sliding. Uncalibrated
    (thresholds staysput.cap_enabled off, docs/D1_MEMO.md note 2) the numbers only, said to change no label. A clip
    with the neighbour test adds neighbour_line before that note (or at the end, calibrated)."""
    line = _staysput_drift_line(mt, calibrated=calibrated)
    near = neighbour_line(mt) if line is not None and mt is not None and mt.verdict != StaysPutVerdict.NO_TRACK else None
    if near is None or line is None:
        return line
    if not calibrated and line.endswith(f"; {STAYSPUT_UNCALIBRATED}"):
        return f"{line[: -len(STAYSPUT_UNCALIBRATED) - 2]}; {near}; {STAYSPUT_UNCALIBRATED}"
    return f"{line}; {near}"


def _staysput_drift_line(mt: MarkTrack | None, *, calibrated: bool = True) -> str | None:
    if mt is None or mt.verdict == StaysPutVerdict.NOT_RUN:
        return None
    if mt.verdict == StaysPutVerdict.NO_TRACK:
        return "no track (SAM did not follow the mark through the clip)"
    numbers = f"drift {_num(mt.drift_rms)}, b1 {_num(mt.b1)}, b2 {_num(mt.b2)}, {'panned' if mt.panned else 'not panned'}"
    if not calibrated:
        return f"{numbers}; {STAYSPUT_UNCALIBRATED}"
    if mt.verdict == StaysPutVerdict.INCONCLUSIVE and not mt.b_pass:
        return "inconclusive (camera did not move enough)"
    if mt.verdict == StaysPutVerdict.INCONCLUSIVE and mt.drift_rms is None:
        return "inconclusive (the whole panel was in view on too few frames)"
    line = f"{numbers}; {mt.verdict.value}"
    if mt.verdict == StaysPutVerdict.INCONCLUSIVE:
        line += f" (between fixed and sliding; sign test {_num(mt.sign_agreement)} on {mt.pairs_n} pairs)"
    if mt.curvature_cap:
        line += "; curvature cap"
    if mt.highlight_cap:
        line += "; highlight cap"
    return line


def note_lines(sheet: SheetModel) -> list[str]:
    """The five lines of section 7 item 6, in the order the sourced cases show works."""
    pickup = next((o for o in sheet.originals if o.role == "pickup"), None)
    camera = pickup.camera_time if pickup else "not present"
    received = (pickup.received_at if pickup else None) or "not recorded"
    hashes = ", ".join(o.sha256[:12] for o in sheet.originals) or "on page 1"
    return [
        f"Start with the dated pickup frames in this sheet: the camera says {camera}, the file was received {received}, "
        f"and the file hashes ({hashes}) are on page 1 and in the sealed manifest.",
        "Ask the company in writing for the full damage report, the repair estimate and every image it holds for this "
        "rental, including the pickup inspection images; Hertz says customers can request more images.",
        "Get an alternative written estimate from a body shop within 10 business days; card network rules compare two "
        "quotes (Visa Core Rules Table 5-19; Amex allows up to 115% of the estimate).",
        "If the company does not answer, escalate to a named executive and file a complaint with your state consumer "
        "protection office; the one sourced renter win came by executive escalation with the company's own scans.",
        "Last, the card dispute, with its clocks: 60 days from the statement under Regulation Z 1026.13, 120 days under "
        "Visa's rules, and the $50 cap of 15 USC 1666i; a dispute does not stop collections, so keep the paper trail.",
    ]


def letter_md(sheet: SheetModel) -> str:
    originals = "\n".join(
        f"- {o.role}: {o.name}, {o.bytes} bytes, sha256 {o.sha256}, camera time {o.camera_time}, received {o.received_at or 'not recorded'}"
        for o in sheet.originals
    ) or "- (no originals recorded)"
    sealed = sheet.clocks.sealed_at or "not sealed"
    return (
        "Write to the company first\n"
        "\n"
        "[date]\n"
        "\n"
        "To: [rental company], damage claims\n"
        f"Re: rental agreement {sheet.agreement_number or '[number]'}, picked up [date], returned [date]\n"
        "\n"
        "I received a damage charge of [amount] for [description]. Before anything else I ask for the full damage report,\n"
        "the written repair estimate and every photograph or scan the company holds for this rental, including the pickup\n"
        "inspection images. Hertz's own support page says customers can request more images.\n"
        "\n"
        "I recorded a walkaround video at pickup and at return. The files were hashed and sealed before any analysis:\n"
        f"{originals}\n"
        f"- sealed at {sealed}\n"
        f"- Walkaround run {sheet.run_id}; the panel-by-panel evidence sheet is attached\n"
        "\n"
        "The attached sheet shows the panel in question at pickup and at return with the marks drawn on both frames. The\n"
        "sheet is evidence support, not legal advice, and I ask that it be read next to your own images.\n"
        "\n"
        "If the charge stands after you have reviewed it, I will obtain an alternative written estimate within 10 business\n"
        "days. Card network rules (Visa Core Rules 13.3 and 13.5; 15 USC 1666i) require that I first try to resolve the\n"
        "charge with you; this letter is that attempt. Please reply within 14 days to [email].\n"
        "\n"
        "[name]\n"
    )


def sheet_stamps(sheet: SheetModel) -> Stamps:
    """The seal's stamps as this sheet read them (method.thresholds["stamps"]); for a model built without that block,
    what page 1 carries."""
    raw = sheet.method.thresholds.get("stamps")
    if isinstance(raw, dict):
        try:
            return Stamps.model_validate(raw)
        except ValueError:
            pass
    tsr = next((o.tsr_time for o in sheet.originals if o.tsr_time), None)
    return Stamps(manifest_ots=_best_ots(*(o.ots_status for o in sheet.originals)), tsr_time=tsr)


def verify_md(sheet: SheetModel) -> str:
    """Appendix C: only the commands whose files this run has (no OTS step without a proof, no openssl step without
    a token), with the token's issuer named."""
    stamps = sheet_stamps(sheet)
    expected = [f"   {o.role}: {o.sha256}" for o in sheet.originals] or ["   (no originals recorded)"]
    steps: list[tuple[str, list[str]]] = [
        ("Hash the video file on your phone or computer and compare it with page 1:",
         ["   shasum -a 256 <your video file>     (Linux: sha256sum <your video file>)", *expected]),
    ]
    if (stamps.manifest_ots or "").startswith(("pending", "confirmed")):
        steps.append(("Read the OpenTimestamps proof (pip install opentimestamps-client):", ["   ots info bundle/manifest.json.ots"]))
        steps.append(("Verify it against the Bitcoin block, or drop the .ots file on https://opentimestamps.org :", ["   ots verify bundle/manifest.json.ots"]))
    if stamps.tsr_time:
        issuer = stamps.tsa_issuer or "the time stamping authority"
        certs, cmd = _TSA_VERIFY.get(stamps.tsa_issuer or "", (
            "that authority's CA certificate and TSA certificate",
            "openssl ts -verify -data bundle/manifest.json -in bundle/manifest.json.tsr -CAfile cacert.pem -untrusted tsa.crt",
        ))
        steps.append((f"Verify the RFC 3161 token from {issuer} ({stamps.tsr_time}) with {certs}:", [f"   {cmd}"]))
    steps.append(("Re-hash every file in the bundle and rebuild the derived files from the stored model answers:", [f"   walkaround verify {sheet.run_id}"]))
    lines = [f"Verify it yourself ({len(steps)} steps; none needs Walkaround's server)", ""]
    for n, (title, body) in enumerate(steps, 1):
        lines += [f"{n}. {title}", *body]
    return "\n".join(lines) + "\n"


def certification_md(sheet: SheetModel) -> str:
    files = "\n".join(f"- {o.role}: {o.name} (sha256 {o.sha256[:12]}...)" for o in sheet.originals) or "- (no originals recorded)"
    return (
        "Renter's certification\n"
        "\n"
        "I, [name], state that:\n"
        "1. I recorded the videos listed below myself, at pickup on [date] and at return on [date], of the vehicle under\n"
        f"   rental agreement {sheet.agreement_number or '[number]'}.\n"
        "2. I uploaded them to Walkaround without editing them, and the sha256 hashes on page 1 match the files on my phone.\n"
        "3. I understand this sheet is evidence support, not legal advice, and that it was produced by code from my videos.\n"
        "\n"
        f"{files}\n"
        "\n"
        "Signed: ______________________   Date: ____________\n"
        "\n"
        "This is a signed statement, not a notarized affidavit.\n"
    )


def glossary_md() -> str:
    return (
        "Glossary\n"
        "\n"
        "already there: the same mark was found at pickup within tau_pre of its return position (rule L3).\n"
        "needs your eye: a human should look; the pair sat in the band between tau_pre and tau_eye, or a cap (K1 to K15) applied (L4).\n"
        "new or unseen: a return mark with no pickup mark near it; new, or not filmed at pickup (L2).\n"
        "not seen at return: a pickup mark with no return mark near it (L1).\n"
        "not comparable: a panel gate failed (G1, G3 to G10); both best frames still show with the reason (L6).\n"
        "view flagged: Muse Spark said a frame shows water, dirt or glare. That is a model's judgment, so it no longer blocks a panel (gate G2 did that until Oct 31). A mark seen in one walk only is capped at needs your eye (K15); a mark in the same place in both walks keeps its label, with a note.\n"
        "units: panel-normalized distance; 1 unit is about 35 cm along a door (the about-cm beside it uses that).\n"
        "tau_pre, tau_eye: the two distance thresholds, calibrated from the run's own repeat frames; printed on the method page.\n"
        "Stays Put: a check that a mark moves with the panel through a short clip; a reflection slides, a scratch stays put.\n"
        "sha256: the fingerprint of a file; one changed byte changes it.\n"
        "sealed: the manifest of hashes was written and offered for time stamping before any model saw the video.\n"
        "config hash: the fingerprint of every threshold the code used; it changes when a threshold changes.\n"
        "rule id: the row of the decision table that produced a label; G gates a panel, K caps a mark, L names a label, E is a claim exit.\n"
    )


# build_sheet ---------------------------------------------------------------------------------------------------------


def _videos(run_id: str) -> dict[str, VideoRecord]:
    out: dict[str, VideoRecord] = {}
    for path in sorted(paths.videos_dir(run_id).glob("*.json")):
        record = read_json(path, VideoRecord)
        out[record.role] = record
    return out


def _ledger_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quantity_key = {"sam_video": ("frames", "frames"), "sam_image": ("images", "images"), "voice": ("audio_seconds", "s")}
    by_kind: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = str(row.get("kind", "?"))
        bucket = by_kind.setdefault(kind, {"calls": 0, "usd": 0.0, "_q": 0.0, "_unit": ""})
        bucket["calls"] += 1
        bucket["usd"] += float(row.get("usd", 0.0) or 0.0)
        if kind == "spark":
            bucket["_q"] += float(row.get("input_tokens", 0) or 0) + float(row.get("output_tokens", 0) or 0)
            bucket["_unit"] = "tokens"
        elif kind in quantity_key:
            key, unit = quantity_key[kind]
            bucket["_q"] += float(row.get(key, 0) or 0)
            bucket["_unit"] = unit
        elif row.get("quantity") is not None:
            bucket["_q"] += float(row["quantity"])
            bucket["_unit"] = str(row.get("unit") or "")
    for bucket in by_kind.values():
        bucket["usd"] = round(bucket["usd"], 4)
        bucket["quantity"] = f"{bucket.pop('_q'):g} {bucket.pop('_unit')}".strip()
    return {"calls": len(rows), "usd": round(sum(b["usd"] for b in by_kind.values()), 4), "by_kind": dict(sorted(by_kind.items()))}


def _mark_box_in_crop(mark: Mark, kf: Keyframe) -> list[float] | None:
    if mark.kf_id == kf.kf_id and mark.bbox_crop:
        return list(mark.bbox_crop)
    if mark.bbox_src and kf.crop:
        c = kf.crop
        x1, y1, x2, y2 = mark.bbox_src
        return [(x1 - c.dx) * c.scale, (y1 - c.dy) * c.scale, (x2 - c.dx) * c.scale, (y2 - c.dy) * c.scale]
    return None


def _draw_marks(bgr: np.ndarray, kf: Keyframe, marks: list[Mark], label_by_mark: dict[str, VerdictLabel]) -> None:
    h, w = bgr.shape[:2]
    thickness = max(2, int(round(w / 600)))
    font_scale = max(0.4, w / 1600)
    for mark in marks:
        box = _mark_box_in_crop(mark, kf)
        if box is None:
            continue
        x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
        x2, y2 = min(w - 1, int(round(box[2]))), min(h - 1, int(round(box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        colour = _LABEL_BGR.get(label_by_mark.get(mark.mark_id), (60, 60, 60))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, thickness)
        tag = f"{mark.mark_id.rsplit('_', 1)[-1]} {mark.type.value} {mark.size_class.value}"
        ty = y1 - 4 if y1 > 14 * font_scale + 4 else y2 + int(14 * font_scale) + 4
        cv2.putText(bgr, tag, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, max(1, thickness - 1), cv2.LINE_AA)


def _cut_crop(rgb: np.ndarray, crop: Crop) -> np.ndarray:
    """The crop's source box resized to (w, h); the same mapping Crop documents."""
    x1, y1 = max(0, int(round(crop.dx))), max(0, int(round(crop.dy)))
    x2 = min(rgb.shape[1], int(round(crop.dx + crop.w / crop.scale)))
    y2 = min(rgb.shape[0], int(round(crop.dy + crop.h / crop.scale)))
    return cv2.resize(rgb[y1:y2, x1:x2], (crop.w, crop.h), interpolation=cv2.INTER_AREA)


def _centre_inside(box: list[float], outer: list[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx < outer[2] and outer[1] <= cy < outer[3]


def _owner_crop(run_id: str, kf: Keyframe, video: VideoRecord | None, tracks: list[PanelTrack], th: dict[str, Any]) -> np.ndarray | None:
    """The keep_plate variant of one crop: the 4K frame from quarantine with every face pixelated and only the rented
    car's plate kept, meaning a plate centred on this keyframe's own panel; a plate elsewhere in the crop belongs to
    a neighbour's car and is pixelated. tracks are this walk's. None when the original is gone, the media lane
    cannot decode, or the keyframe has no crop geometry."""
    if kf.crop is None:
        return None
    original = paths.quarantine_original(run_id, kf.role)
    if not original.is_file():
        return None
    try:
        from . import video as video_mod

        frame = video_mod.read_frame_at(original, kf.pts)
    except Exception:  # noqa: BLE001 - a stub or a decode failure: no owner frame from this original
        return None
    face_boxes = [list(b) for b in kf.blur.face_boxes]
    plate_boxes = [list(b) for b in kf.blur.plate_boxes]
    if tracks and video is not None and video.proxy is not None:
        common: dict[str, Any] = dict(
            fps=video.proxy.fps,
            media_size_proxy=(video.proxy.width, video.proxy.height),
            media_size_src=(int(frame.shape[1]), int(frame.shape[0])),
            th=th,
        )
        face_boxes += privacy.track_boxes_at(kf.f_proxy, tracks, nouns=(Noun.FACE,), **common)
        plate_boxes += privacy.track_boxes_at(kf.f_proxy, tracks, nouns=(Noun.LICENSE_PLATE,), **common)
    # TODO(D1): on the real keep_plate walk, check the rented car's plate is centred in its bumper or trunk box and
    # a neighbour's plate in a door crop is not; widen the test to the car box if a mounted plate gets pixelated.
    neighbours = [b for b in plate_boxes if not _centre_inside(b, list(kf.box_src))]
    blurred = privacy.blur_frame(frame, track_boxes_src=face_boxes + neighbours, detections=[], keep_plate_region=True, th=th)
    return _cut_crop(blurred, kf.crop)


def _write_frame(run_id: str, kf: Keyframe, marks: list[Mark], label_by_mark: dict[str, VerdictLabel]) -> str | None:
    """sheet/frames/<kf_id>.jpg: the blurred crop with the marks drawn (the blurred car-box frame, undrawn, when the
    crop is missing); None without pixels."""
    root = paths.run_dir(run_id)
    src = root / kf.crop.path if kf.crop and kf.crop.path else None
    draw = src is not None and src.is_file()
    if not draw:
        src = paths.keyframe_frame_path(run_id, kf.role, kf.kf_id)
        if not src.is_file():
            return None
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    if draw:
        _draw_marks(bgr, kf, marks, label_by_mark)
    dst = paths.sheet_dir(run_id) / "frames" / f"{kf.kf_id}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return paths.rel(run_id, dst)


def owner_frame_file(run_id: str, kf_id: str) -> Path:
    """The keep_plate frame of one keyframe; served to the run owner only (privacy.served_file_allowed)."""
    return paths.sheet_dir(run_id) / "frames" / f"{kf_id}.owner.jpg"


def _write_owner_frame(run_id: str, kf: Keyframe, marks: list[Mark], label_by_mark: dict[str, VerdictLabel], video: VideoRecord | None, tracks: list[PanelTrack], th: dict[str, Any]) -> str | None:
    """sheet/frames/<kf_id>.owner.jpg with the marks drawn: written while the quarantined original exists and kept as
    it is once the original is deleted at run end, so a rebuild or a resumed sheet prints the same frames. None when
    no owner frame could ever be made."""
    dst = owner_frame_file(run_id, kf.kf_id)
    rgb = _owner_crop(run_id, kf, video, tracks, th)
    if rgb is not None:
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        _draw_marks(bgr, kf, marks, label_by_mark)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return paths.rel(run_id, dst) if dst.is_file() else None


def _coverage_lines(role: str, states: dict[PanelId, Any]) -> list[str]:
    lines = []
    for pid in PanelId:
        cov = states.get(pid)
        if cov is None:
            continue
        line = f"{display_name(pid)}: {cov.state.value}"
        if cov.seconds:
            line += f", {cov.seconds:.1f} s"
        lines.append(line)
    return lines


def _storage_line(run: RunManifest, clocks: DeletionClocks) -> str:
    if run.responses_store == ResponsesStore.FALSE_ACCEPTED:
        text = "Model requests were sent with storage off and the models accepted it; nothing was kept after the run."
    elif run.responses_store == ResponsesStore.DELETED_PER_REQUEST:
        text = "A model refused storage off, so every stored model response was deleted by Walkaround when the run ended."
    else:
        text = "Model requests were sent with storage off; whether every model honoured it was not confirmed for this run."
    text += f" Files API upload deleted: {clocks.files_deleted_at or 'not yet'}; quarantine deleted: {clocks.quarantine_deleted_at or 'not yet'}."
    if not (clocks.files_deleted_at and clocks.quarantine_deleted_at):
        text += " " + RUN_END_NOTE
    return text


def _deletions_at_sheet_time(rows: list[DeletionRecord], events: list[Event], head_hash: str | None) -> list[DeletionRecord]:
    """The deletion rows this sheet prints: announced on the chain up to the frozen head (blur.done for the unblurred
    proxy, files.deleted, responses.deleted, quarantine.deleted) or timestamped before the head event. Rows written
    after the sheet (a keep_plate original deleted at close, a late top-up) stay in deletions.json and
    manifest_run.json, so a rebuild after close prints the same sheet. Without a frozen head every row counts."""
    head = events[-1] if events and head_hash and events[-1].hash == head_hash else None
    if head is None:
        return list(rows)
    announced: set[str] = set()
    for event in events:
        payload = event.payload
        if event.kind == EventKind.FILES_DELETED and isinstance(payload.get("file_id"), str):
            announced.add(payload["file_id"])
        elif event.kind == EventKind.RESPONSES_DELETED and isinstance(payload.get("request_id"), str):
            announced.add(payload["request_id"])
        elif event.kind == EventKind.QUARANTINE_DELETED and isinstance(payload.get("files"), list):
            announced.update(str(f) for f in payload["files"])
        elif event.kind == EventKind.BLUR_DONE and isinstance(payload.get("role"), str):
            announced.add(f"media/{payload['role']}.proxy.mp4")  # paths.proxy_path, deleted at blur.done
    head_at = seal._parse_iso(head.ts)
    return [r for r in rows if r.target in announced or _before(r.deleted_at, head_at)]


def _sheet_time_problems(run_id: str, run: RunManifest) -> list[str]:
    """What the sheet can check about its own inputs without depending on later state: every file manifest.json
    lists that is still present re-hashes to its entry, and the chain verifies from seq 0 to the head this sheet
    prints. An original deleted at run end (or left out of a rebuild copy) is skipped here; walkaround verify checks
    it against manifest_run.json's deletions."""
    manifest_path = paths.manifest_json(run_id)
    if not manifest_path.is_file():
        return ["bundle/manifest.json missing"]
    manifest = read_json(manifest_path, SealManifest)
    present = seal.verify_manifest(manifest, paths.run_dir(run_id), expected_missing={e.path for e in manifest.files})
    return present + seal.chain_problems_through(run_id, seal.sha256_file(manifest_path), head_hash=run.events_head_hash)


def build_sheet(run_id: str, *, th: dict[str, Any]) -> SheetModel:
    """Fill SheetModel from run.json, videos/*.json, bundle/manifest.json, derived/*.json, deletions.json, the event
    log up to the frozen head and the run's ledger rows. Never recomputes a verdict."""
    run = read_json(paths.run_json(run_id), RunManifest)
    videos = _videos(run_id)
    events = seal.events_through(run_id, run.events_head_hash)
    stamps = seal.stamps_from_events(events)
    panels: list[Panel] = _read_list_opt(paths.panels_json(run_id), Panel)
    keyframes: list[Keyframe] = _read_list_opt(paths.keyframes_json(run_id), Keyframe)
    marks: list[Mark] = _read_list_opt(paths.marks_json(run_id), Mark)
    mark_tracks: list[MarkTrack] = _read_list_opt(paths.mark_tracks_json(run_id), MarkTrack)
    matches: list[PanelMatch] = _read_list_opt(paths.matches_json(run_id), PanelMatch)
    verdicts: list[Verdict] = _read_list_opt(paths.verdicts_json(run_id), Verdict)
    coverage: Coverage | None = _read_opt(paths.coverage_json(run_id), Coverage)
    narration: Narration | None = _read_opt(paths.narration_json(run_id), Narration)
    calibration: Calibration | None = _read_opt(paths.calibration_json(run_id), Calibration)
    claim: Claim | None = _read_opt(paths.claim_json(run_id), Claim)
    deletions: Deletions | None = _read_opt(paths.deletions_json(run_id), Deletions)
    tracks: list[PanelTrack] = _read_list_opt(paths.tracks_json(run_id), PanelTrack) if run.keep_plate else []
    rows = [r for r in ledger.rows(paths.ledger_path()) if r.get("run_id") == run_id]

    # Page 1: originals and clocks.
    originals: list[SheetOriginal] = []
    camera_by_role: dict[str, str] = {}
    for role in sorted(videos, key=_role_order):
        v = videos[role]
        sha = v.sha256_server or v.sha256_browser
        camera_by_role[role] = v.ffprobe.creation_time if v.ffprobe else seal.NOT_PRESENT
        if sha is None:
            continue
        ots = _best_ots({"pickup": stamps.pickup_ots, "return": stamps.return_ots}.get(role), stamps.manifest_ots)
        originals.append(
            SheetOriginal(
                role=role, name=v.original_filename, bytes=v.bytes, sha256=sha, camera_time=camera_by_role[role],
                received_at=v.received_at, hash_match=v.hash_match, ots_status=ots, tsr_time=stamps.tsr_time,
                bitcoin_block=_block_of(ots),
            )
        )
    deletion_rows = _deletions_at_sheet_time(deletions.rows if deletions else [], events, run.events_head_hash)
    clocks = DeletionClocks(
        files_deleted_at=max((d.deleted_at for d in deletion_rows if d.kind == DeletionKind.FILES_API and d.api_response is not None), default=None),
        responses_deleted_at=max((d.deleted_at for d in deletion_rows if d.kind == DeletionKind.RESPONSE), default=None),
        quarantine_deleted_at=max((d.deleted_at for d in deletion_rows if d.kind == DeletionKind.QUARANTINE), default=None),
    )

    # Panel blocks.
    kf_by_id = {k.kf_id: k for k in keyframes}
    match_by_panel = {m.panel_id: m for m in matches}
    marks_by_id = {m.mark_id: m for m in marks}
    mt_by_mark = {t.mark_id: t for t in mark_tracks}
    staysput_calibrated = bool(th["staysput"].get("cap_enabled", True))  # match.staysput_caps reads the same leaf
    label_by_mark = {mid: v.label for v in verdicts for mid in v.mark_ids}
    panel_by_id = {p.panel_id: p for p in panels}
    owner_frames = 0

    def chosen_keyframe(panel_id: PanelId, side: str) -> Keyframe | None:
        pm = match_by_panel.get(panel_id)
        if pm is not None and pm.kf_pair:
            kid = pm.kf_pair[0 if side == "pickup" else 1]
            if kid in kf_by_id:
                return kf_by_id[kid]
        candidates = [k for k in keyframes if k.panel_id == panel_id and _side(k.role) == side]
        return max(candidates, key=lambda k: k.score.total, default=None)

    def frame_for(panel_id: PanelId, side: str) -> SheetFrame | None:
        nonlocal owner_frames
        kf = chosen_keyframe(panel_id, side)
        if kf is None:
            return None
        panel_marks = [m for m in marks if m.panel_id == panel_id and _side(m.role) == side]
        path = _write_frame(run_id, kf, panel_marks, label_by_mark) or ""
        if run.keep_plate:
            walk_tracks = [t for t in tracks if t.role == kf.role]
            owner = _write_owner_frame(run_id, kf, panel_marks, label_by_mark, videos.get(kf.role), walk_tracks, th)
            if owner:
                path = owner
                owner_frames += 1
        return SheetFrame(
            kf_id=kf.kf_id, path=path, video_time_s=kf.pts,
            wall_clock=_wall_clock(camera_by_role.get(kf.role, seal.NOT_PRESENT), kf.pts),
            mark_ids=[m.mark_id for m in panel_marks],
        )

    blocks: list[SheetPanelBlock] = []
    for pid in PanelId:
        cov_seconds: dict[str, float] = {}
        cov_states: dict[str, Any] = {}
        if coverage is not None:
            for role, per_panel in coverage.per_role.items():
                if pid in per_panel:
                    cov_seconds[role] = per_panel[pid].seconds
                    cov_states[role] = per_panel[pid].state
        elif pid in panel_by_id:
            for role, pr in panel_by_id[pid].per_role.items():
                cov_seconds[role] = pr.seconds_visible
                cov_states[role] = pr.state
        lines: list[SheetVerdictLine] = []
        for v in [v for v in verdicts if v.panel_id == pid]:
            first = next((marks_by_id[i] for i in v.mark_ids if i in marks_by_id), None)
            sp = next((s for s in (staysput_line(mt_by_mark[i], calibrated=staysput_calibrated) for i in v.mark_ids if i in mt_by_mark) if s), None)
            size_text = size_line(v.size) if v.size is not None or v.rule_id != "L6" else None  # not comparable: no size, no ruler line
            extra = threshold_line(v.size, th)
            lines.append(
                SheetVerdictLine(
                    verdict_id=v.verdict_id, label=v.label, rule_ids=_dedup([v.rule_id, *v.trace.rule_ids]), mark_ids=list(v.mark_ids),
                    type=first.type if first else None, size_class=first.size_class if first else None,
                    distance_units=v.distance_units, distance_cm_est=v.distance_cm_est, staysput_line=sp,
                    size_line=f"{size_text}; {extra}" if extra else size_text,
                    ordinary_wear_line=ordinary_wear_line(first.type if first else None, pid),
                    view_line=v.view_note,
                )
            )
        chips: list[SheetChip] = []
        if narration is not None:
            for role, nr in sorted(narration.per_role.items(), key=lambda kv: _role_order(kv[0])):
                chips += [SheetChip(turn_id=t.turn_id, start_ms=t.start_ms, end_ms=t.end_ms, text=t.text, chip_path=t.chip_path) for t in nr.turns if t.pinned_panel == pid]
        pm = match_by_panel.get(pid)
        blocks.append(
            SheetPanelBlock(
                panel_id=pid, display_name=display_name(pid), lead_with_frames=pid in WHEEL_PANELS or pid in LOWER_BUMPERS,
                pickup=frame_for(pid, "pickup"), return_=frame_for(pid, "return"),
                coverage_seconds=dict(sorted(cov_seconds.items(), key=lambda kv: _role_order(kv[0]))),
                coverage_states=dict(sorted(cov_states.items(), key=lambda kv: _role_order(kv[0]))),
                verdicts=lines, group_lines=[g.text for g in pm.groups] if pm else [], chips=chips,
            )
        )
    if coverage is not None and coverage.odometer.frame_path:
        target = next((b for b in blocks if b.pickup or b.return_), blocks[0])
        target.odometer_path = coverage.odometer.frame_path

    not_comparable: list[SheetNotComparable] = []
    for pid in PanelId:
        pm = match_by_panel.get(pid)
        ids = list(pm.reasons) if pm is not None and pm.state.value == "not_comparable" else []
        if not ids and pid in panel_by_id and not panel_by_id[pid].comparable and panel_by_id[pid].not_comparable_reasons:
            ids = list(panel_by_id[pid].not_comparable_reasons)
        if ids:
            not_comparable.append(SheetNotComparable(panel_id=pid, rule_ids=ids, reasons=[rules.describe([i])[0] if i in rules.RULES else i for i in ids]))

    # Coverage summary per walk.
    coverage_summary: dict[str, list[str]] = {}
    if coverage is not None:
        for role in sorted(coverage.per_role, key=_role_order):
            coverage_summary[role] = _coverage_lines(role, coverage.per_role[role])
        if coverage.missing and "pickup" in coverage_summary:
            coverage_summary["pickup"].append("missing at pickup: " + ", ".join(display_name(p) for p in coverage.missing))
        if coverage.topup_text and "pickup" in coverage_summary:
            coverage_summary["pickup"].append(f"top-up asked: {coverage.topup_text}")
    elif panels:
        for role in sorted({r for p in panels for r in p.per_role}, key=_role_order):
            coverage_summary[role] = _coverage_lines(role, {p.panel_id: p.per_role[role] for p in panels if role in p.per_role})

    # Privacy and the plate line. A walk with keyframes answers from their image checks; one without falls back to
    # the plate track lane, read here only when needed (tracks.json carries every per-frame row).
    plate_found: dict[str, bool] = {}
    for role in videos:
        kfs = [k for k in keyframes if k.role == role]
        if kfs:
            plate_found[role] = any(k.blur.plate_boxes for k in kfs)
            continue
        if not tracks and paths.tracks_json(run_id).is_file():
            tracks = read_json_list(paths.tracks_json(run_id), PanelTrack)
        if tracks:
            plate_found[role] = any(t.role == role and t.noun == Noun.LICENSE_PLATE for t in tracks)
    plate_line = "plate kept on this sheet at the renter's request at upload" if run.keep_plate and owner_frames else "plate blurred"
    blurred_lines = []
    for role in sorted(videos, key=_role_order):
        kfs = [k for k in keyframes if k.role == role]
        if kfs:
            n_plate = sum(1 for k in kfs if k.blur.plate_boxes)
            n_face = sum(1 for k in kfs if k.blur.face_boxes)
            blurred_lines.append(f"{role}: SAM image checks on {len(kfs)} keyframes found a plate on {n_plate} and faces on {n_face}; the display proxy was pixelated from the plate and face tracks")
        else:
            blurred_lines.append(f"{role}: no keyframes recorded; the display proxy was pixelated from the plate and face tracks")
    if run.keep_plate:
        blurred_lines.append("keep_plate was chosen at upload: the plate stays only in the run owner's sheet frames" + ("" if owner_frames else " (none could be produced for this sheet, so every frame here is blurred)"))

    thresholds = dict(run.thresholds or th)
    thresholds["calibration"] = calibration.model_dump(mode="json") if calibration else None
    thresholds["ledger_totals"] = _ledger_totals(rows)
    thresholds["stamps"] = stamps.model_dump(mode="json")  # carried here until SheetModel has a stamps field
    problems = _sheet_time_problems(run_id, run)
    method = SheetMethod(
        config_hash=run.config_hash, thresholds=thresholds, eval_lines=[], blurred=blurred_lines, deletions=list(deletion_rows),
        storage_line=_storage_line(run, clocks),
        # The rebuild comparison runs after this sheet exists (walkaround verify). At sheet time the chain up to the
        # frozen head can be checked, plus whichever originals are still in quarantine (none, once run end has
        # deleted them), and the wording claims no more than that, the same way before and after run end.
        rebuild_status=(
            "event chain verified to the head on page 1 and any original still in quarantine re-hashed at sheet time; "
            "run walkaround verify to compare a rebuild"
            if not problems
            else f"verify at sheet time: {problems[0]}"
        ),
        what_seal_does_not_prove=list(WHAT_SEAL_DOES_NOT_PROVE), hertz_threshold_note=HERTZ_NOTE,
    )

    sheet = SheetModel(
        run_id=run_id, app_commit=run.app_commit, config_hash=run.config_hash, events_head_hash=run.events_head_hash,
        clocks=run.clocks, deletion_clocks=clocks, originals=originals, plate_line=plate_line,
        privacy_line=privacy.privacy_line(run, plate_found, deletion_rows, baseten=_used_baseten(run_id)), coverage_summary=coverage_summary, panels=blocks,
        stand_in_line=stand_in_line(run),
        not_comparable=not_comparable, method=method, footer_hash_prefixes=[o.sha256[:12] for o in originals],
    )
    sheet.note_lines = note_lines(sheet)
    appendices = {"A certification": certification_md(sheet), "B glossary": glossary_md(), "C verify it yourself": verify_md(sheet)}
    if claim is not None:
        appendices["D claim check"] = _claim_text(claim, th)
    sheet.appendices = appendices
    sheet.letter_md = letter_md(sheet)
    return sheet


def stand_in_line(run: RunManifest) -> str:
    """Page 1 says so when the lanes were sent with stand-in prompts: a test clip, not a car walk."""
    if not run.prompts:
        return ""
    sent = "; ".join(f"{noun} sent as '{text}'" if text else f"{noun}: no request, the whole frame stands in" for noun, text in sorted(run.prompts.items()))
    return f"Test run on a clip with no car: lanes were sent with stand-in prompts ({sent}). Panel names below are slots, not car parts."


def _claim_text(claim: Claim, th: dict[str, Any]) -> str:
    lines = ["Claim check", ""]
    if claim.exit is not None:
        r = rules.rule(claim.exit.value)
        lines.append(f"exit {r.id} {r.label}: {r.text}")
    if claim.panel_id is not None:
        lines.append(f"panel: {display_name(claim.panel_id)}" + (f" (side from {claim.side_source})" if claim.side_source else ""))
    if claim.coverage_state is not None:
        lines.append(f"coverage at pickup: {claim.coverage_state.value}")
    if claim.paragraph:
        lines += ["", claim.paragraph]  # it already holds the size sentence when the claim was compared
    elif claim.size is not None:
        size_text = size_line(claim.size)
        extra = threshold_line(claim.size, th)
        lines += ["", f"{size_text}; {extra}" if extra else size_text]
    return "\n".join(lines) + "\n"


def _used_baseten(run_id: str) -> bool:
    """The run sent second looks to Muse Spark Open on Baseten: a look answered there, or a provenance line exists."""
    log = paths.open_weights_log(run_id)
    if log.is_file() and log.stat().st_size > 0:
        return True
    looks = paths.looks_json(run_id)
    if not looks.is_file() or looks.stat().st_size == 0:
        return False
    return any(row.get("engine") == "baseten" for row in json.loads(looks.read_text(encoding="utf-8")))


# Rendering -----------------------------------------------------------------------------------------------------------


def _flatten(doc: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Nested threshold dict -> [("panels.crop_pad_frac", "0.1"), ...] in key order."""
    if isinstance(doc, dict):
        out: list[tuple[str, str]] = []
        for key in sorted(doc):
            out += _flatten(doc[key], f"{prefix}{key}.")
        return out
    return [(prefix.rstrip("."), doc if isinstance(doc, str) else repr(doc))]


def _has_page_content(block: SheetPanelBlock) -> bool:
    """A panel gets its own page when it has a frame, a verdict, a group line or a narration chip."""
    return bool(block.pickup or block.return_ or block.verdicts or block.group_lines or block.chips)


def _asset_src(inline_root: Path | None) -> Callable[[str], str]:
    """img src for a run-relative path: the file inlined as a data URL when it sits under inline_root, so the page
    works wherever it is opened (served at /api/runs/<id>/sheet, saved alone, printed); else ../<path>, which
    resolves from sheet/sheet.html on disk."""
    base = Path(inline_root).resolve() if inline_root is not None else None

    def src(rel: str) -> str:
        if base is not None and rel:
            path = (base / rel).resolve()
            if path.is_relative_to(base) and path.is_file() and path.suffix.lower() in _MIME:
                return f"data:{_MIME[path.suffix.lower()]};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        return f"../{rel}"

    return src


def render_html(sheet: SheetModel, *, template_dir: Path = TEMPLATE_DIR, inline_root: Path | None = None) -> str:
    """Jinja2 to one self-contained HTML page set: no external assets, ASCII only, footer on every page. With
    inline_root (the run dir) the frames are embedded as data URLs."""
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape(["html"]), undefined=StrictUndefined)
    thresholds = {k: v for k, v in sheet.method.thresholds.items() if k not in ("calibration", "ledger_totals", "stamps")}
    clocks = sheet.deletion_clocks
    context = {
        "sheet": sheet,
        "asset_src": _asset_src(inline_root),
        "ots_note": ots_note,
        "tsa_issuer": sheet_stamps(sheet).tsa_issuer,
        "run_end_note": "" if clocks.files_deleted_at and clocks.quarantine_deleted_at else RUN_END_NOTE,
        "panel_pages": [b for b in sheet.panels if _has_page_content(b)],
        "frameless_blocks": [b for b in sheet.panels if not _has_page_content(b)],
        "label_text": LABEL_TEXT,
        "rule_text": {r.id: f"{r.label}: {r.text}" for r in rules.RULES.values()},
        "block_rule_ids": {b.panel_id.value: _dedup([rid for v in b.verdicts for rid in v.rule_ids]) for b in sheet.panels},
        "panel_names": {p.value: display_name(p) for p in PanelId},
        "flat_thresholds": _flatten(thresholds),
    }
    return _ascii(env.get_template("sheet.html").render(**context))


_PDF_DATES = re.compile(rb"(/(?:CreationDate|ModDate) \(D:)\d{14}")


def pin_pdf_dates(pdf_path: Path, stamp: str) -> None:
    """Chromium writes the print time into /CreationDate and /ModDate, the only bytes that differ between two prints of
    one page (measured 10/31). Both become `stamp` (14 digits, same width), so the xref offsets hold and a reprint of
    the same sheet is byte-identical to the one the run manifest hashed."""
    data = Path(pdf_path).read_bytes()
    Path(pdf_path).write_bytes(_PDF_DATES.sub(lambda m: m.group(1) + stamp.encode("ascii"), data))


def pdf_stamp(sheet: SheetModel) -> str:
    """YYYYMMDDHHMMSS (UTC) of the seal, the time the sheet is about; a run that was never sealed gets 2000-01-01."""
    sealed = sheet.clocks.sealed_at or ""
    digits = re.sub(r"\D", "", sealed)[:14]
    return digits if len(digits) == 14 else "20000101000000"


def render_pdf(html_path: Path, pdf_path: Path, *, timeout_s: int = 60, stamp: str | None = None) -> Path:
    """Playwright Chromium prints the HTML; raises TimeoutError past timeout_s and RuntimeError without Playwright.
    With stamp, the PDF's two dates are pinned to it (pin_pdf_dates)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is not installed; the HTML sheet stands and the PDF button says 'PDF retry'") from exc

    def _print() -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=timeout_s * 1000)
                page.pdf(path=str(pdf_path), format="Letter", print_background=True, prefer_css_page_size=True)
            finally:
                browser.close()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(_print).result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"PDF render exceeded {timeout_s} s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if stamp is not None:
        pin_pdf_dates(Path(pdf_path), stamp)
    return Path(pdf_path)


def _write_text(path: Path, text: str) -> str:
    data = _ascii(text).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_sheet(run_id: str, *, th: dict[str, Any]) -> tuple[SheetModel, dict[str, str]]:
    """sheet.json, sheet.html, the bundle texts (note.md, letter.md, verify.md) and evidence.pdf when Playwright is
    present; returns the model and {run-relative file: sha256}. A PDF failure leaves the HTML standing."""
    sheet = build_sheet(run_id, th=th)
    hashes = {"sheet/sheet.json": write_json(paths.sheet_json(run_id), sheet)}
    html_path = paths.sheet_html(run_id)
    hashes["sheet/sheet.html"] = _write_text(html_path, render_html(sheet, inline_root=paths.run_dir(run_id)))
    hashes["bundle/letter.md"] = _write_text(paths.bundle_file(run_id, "letter.md"), sheet.letter_md)
    hashes["bundle/verify.md"] = _write_text(paths.bundle_file(run_id, "verify.md"), sheet.appendices.get("C verify it yourself", ""))
    hashes["bundle/note.md"] = _write_text(paths.bundle_file(run_id, "note.md"), "\n".join(f"{i}. {line}" for i, line in enumerate(sheet.note_lines, 1)) + "\n")
    pdf_path = paths.sheet_pdf(run_id)
    try:
        render_pdf(html_path, pdf_path, timeout_s=int(th.get("orchestrator", {}).get("pdf_timeout_s", 60)), stamp=pdf_stamp(sheet))
        hashes["sheet/evidence.pdf"] = seal.sha256_file(pdf_path)
    except (RuntimeError, TimeoutError, OSError):
        pdf_path.unlink(missing_ok=True)
    return sheet, hashes
