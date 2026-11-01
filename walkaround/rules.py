"""The rule id table of docs/ARCHITECTURE.md section 6 plus the claim exits of 5.11.

Every verdict prints its rule ids beside its label. G rules gate a panel (not_comparable), K flags cap a mark
(lower to needs_your_eye), L rules name the band a distance fell in, E rules are claim-first exits. No code here
decides anything; match.verdict() and the claim module import these ids so every printed id exists in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    text: str
    source: str  # section of docs/ARCHITECTURE.md


GATES: tuple[Rule, ...] = (
    Rule("G1", "not filmed", "the panel is not 'filmed' in either walk", "6; states 5.4"),
    Rule("G2", "crop condition (retired)", "wet, dirty or glare flag from Spark on either crop; no longer a gate since Oct 31, a flagged view caps instead (K15)", "6; marks_v1 conditions; retired by the lead 10/31"),
    Rule("G3", "blurry keyframe", "sharpness below 0.4 of the candidates' max on either keyframe", "6; 5.3"),
    Rule("G4", "mask IoU", "whitened mask IoU of the paired keyframes under 0.7", "6; 5.8"),
    Rule("G5", "handle self-test", "the door handle more than tau_pre apart between walks", "6; 5.8"),
    Rule("G6", "side uncertain", "side_uncertain, or front_sign null, on either walk", "6; 5.2"),
    Rule("G7", "track incomplete", "the noun was not run or the track was incomplete in either walk (a zero-match lane is not_filmed, G1, never G7)", "6; 5.1"),
    Rule("G8", "hood IoU", "hood pair with whitened mask IoU under 0.8", "6; 5.5"),
    Rule("G9", "mask definition differs", "window_subtracted differs between the paired keyframes", "6; 5.1"),
    Rule("G10", "marks incomplete", "Spark marks truncated after the retry on every keyframe of the panel in either walk", "6; 5 step 8"),
)

CAPS: tuple[Rule, ...] = (
    Rule("K1", "in the band", "d between tau_pre_r and tau_eye_r", "6; 5.8"),
    Rule("K2", "single frame", "the mark appears in 1 of 3 keyframes while the other two were complete and sharp", "6; 5.6"),
    Rule("K3", "tiny mark", "size under 0.03 units or bbox under 12 px", "6; 5.7"),
    Rule("K4", "uncertain type", "type dirt, reflection or unsure", "6; marks_v1"),
    Rule("K5", "incompatible types", "the paired marks have incompatible types", "6; 5.8"),
    Rule("K6", "stays put", "Stays Put sliding or inconclusive, or the curvature or highlight cap", "6; 5.9"),
    Rule("K7", "second look", "the second look says different or unsure", "6; 5 step 10"),
    Rule("K8", "count mode only", "the mark came from count mode only (source spark_count)", "6; 5.7"),
    Rule("K9", "droplets or spray", "droplet-like or road-spray-like panel", "6; 5.6"),
    Rule("K10", "one-mark wheel", "one-mark wheel outside the radius, type and size rule", "6; 5.8"),
    Rule("K11", "claim region", "claim ROI is a region (E7) or the claim crop carries a condition flag (E8)", "6; 5.11"),
    Rule("K12", "low confidence", "Spark confidence low on either mark", "6; marks_v1"),
    Rule("K13", "other walk truncated", "one of the two keyframes has marks_status truncated", "6; 5 step 8"),
    Rule("K14", "named by the renter", "Muse Spark boxed the mark only because the renter named it; seed_v1 gives no confidence or size, so its type or size may be the renter's words", "6; 5 step 11"),
    Rule("K15", "view flagged", "Muse Spark flagged the pickup or return view of the panel wet, dirty or glare, so a mark seen in one walk only may be dirt, water or a reflection, or may have been hidden in the other walk; a mark in the same place in both walks keeps its label with a note", "6; marks_v1 conditions; replaces gate G2, lead 10/31"),
)

LABELS: tuple[Rule, ...] = (
    Rule("L1", "not seen at return", "d is None for a pickup mark: not_seen_at_return", "6, labels paragraph"),
    Rule("L2", "new or unseen", "d is None for a return mark: new_or_unseen", "6, labels paragraph"),
    Rule("L3", "already there", "d < tau_pre_r", "6, labels paragraph"),
    Rule("L4", "needs your eye", "tau_pre_r <= d < tau_eye_r, or any K flag lowered the label", "6, labels paragraph"),
    Rule("L5", "beyond the band", "d >= tau_eye_r: both marks keep their unmatched labels", "6, labels paragraph"),
    Rule("L6", "not comparable", "a G rule failed for the panel; both best frames still show", "6, panel gates"),
    Rule("L7", "outline changed in place", "G4 alone failed on every keyframe pair tried, the best whitened mask IoU is from 0.5 to under 0.7, the two keyframes' box aspect ratios agree within 2% and the largest blob of the mask difference holds at least half of it: the panel may be reshaped, so needs your eye; never already there or new", "6, panel gates; lead 11/1"),  # noqa: E501
)

EXITS: tuple[Rule, ...] = (
    Rule("E1", "no car panel", "the claim image shows no car panel", "5.11"),
    Rule("E2", "naming disagreed", "Spark and the SAM vote name different panels: two candidates and one tap, recorded as the renter's input", "5.11"),
    Rule("E2b", "side unknown", "no side from the text or Spark: both sides shown", "5.11"),
    Rule("E3", "never filmed", "the named panel was not filmed at pickup", "5.11"),
    Rule("E4", "only glimpsed", "the named panel was glimpsed, not filmed: the sharpest frame and the seconds show", "5.11"),
    Rule("E5", "cannot place", "the mask touches the border and the landmark fit is off or has under 3 landmarks", "5.11"),
    Rule("E6", "no marks in ROI", "marks_v1 found nothing in the padded ROI", "5.11"),
    Rule("E7", "drawn box is a region", "the drawn box covers over 40% of the panel", "5.11"),
    Rule("E8", "claim crop condition", "glare, wet or dirt on the claim crop: capped at needs your eye", "5.11"),
)

RULES: dict[str, Rule] = {r.id: r for r in (*GATES, *CAPS, *LABELS, *EXITS)}


def rule(rule_id: str) -> Rule:
    """The rule for an id; KeyError on an unknown id so no sheet prints an id that is not in the table."""
    return RULES[rule_id]


def gate_ids() -> list[str]:
    return [r.id for r in GATES]


def cap_ids() -> list[str]:
    return [r.id for r in CAPS]


def label_ids() -> list[str]:
    return [r.id for r in LABELS]


def exit_ids() -> list[str]:
    return [r.id for r in EXITS]


def describe(rule_ids: list[str]) -> list[str]:
    """'K2 single frame: the mark appears in ...' per id, in the order given."""
    return [f"{r.id} {r.label}: {r.text}" for r in (rule(i) for i in rule_ids)]
