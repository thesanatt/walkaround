// Mirror of walkaround/rules.py (generated from it on 2026-10-31, K15 and the retired G2 added that night) so every printed rule id carries its meaning on hover.
// If rules.py changes, regenerate: the id set must match, or a sheet id would print here without its text.
export interface RuleText {
  label: string;
  text: string;
}

export const RULES: Record<string, RuleText> = {
  G1: { label: "not filmed", text: "the panel is not 'filmed' in either walk" },
  G2: { label: "crop condition (retired)", text: "wet, dirty or glare flag from Spark on either crop; no longer a gate since Oct 31, a flagged view caps instead (K15)" },
  G3: { label: "blurry keyframe", text: "sharpness below 0.4 of the candidates' max on either keyframe" },
  G4: { label: "mask IoU", text: "whitened mask IoU of the paired keyframes under 0.7" },
  G5: { label: "handle self-test", text: "the door handle more than tau_pre apart between walks" },
  G6: { label: "side uncertain", text: "side_uncertain, or front_sign null, on either walk" },
  G7: { label: "track incomplete", text: "the noun was not run or the track was incomplete in either walk (a zero-match lane is not_filmed, G1, never G7)" },
  G8: { label: "hood IoU", text: "hood pair with whitened mask IoU under 0.8" },
  G9: { label: "mask definition differs", text: "window_subtracted differs between the paired keyframes" },
  G10: { label: "marks incomplete", text: "Spark marks truncated after the retry on every keyframe of the panel in either walk" },
  K1: { label: "in the band", text: "d between tau_pre_r and tau_eye_r" },
  K2: { label: "single frame", text: "the mark appears in 1 of 3 keyframes while the other two were complete and sharp" },
  K3: { label: "tiny mark", text: "size under 0.03 units or bbox under 12 px" },
  K4: { label: "uncertain type", text: "type dirt, reflection or unsure" },
  K5: { label: "incompatible types", text: "the paired marks have incompatible types" },
  K6: { label: "stays put", text: "Stays Put sliding or inconclusive, or the curvature or highlight cap" },
  K7: { label: "second look", text: "the second look says different or unsure" },
  K8: { label: "count mode only", text: "the mark came from count mode only (source spark_count)" },
  K9: { label: "droplets or spray", text: "droplet-like or road-spray-like panel" },
  K10: { label: "one-mark wheel", text: "one-mark wheel outside the radius, type and size rule" },
  K11: { label: "claim region", text: "claim ROI is a region (E7) or the claim crop carries a condition flag (E8)" },
  K12: { label: "low confidence", text: "Spark confidence low on either mark" },
  K13: { label: "other walk truncated", text: "one of the two keyframes has marks_status truncated" },
  K14: { label: "named by the renter", text: "Muse Spark boxed the mark only because the renter named it; seed_v1 gives no confidence or size, so its type or size may be the renter's words" },
  K15: { label: "view flagged", text: "Muse Spark flagged the pickup or return view of the panel wet, dirty or glare, so a mark seen in one walk only may be dirt, water or a reflection, or may have been hidden in the other walk; a mark in the same place in both walks keeps its label with a note" },
  K16: { label: "the other walk may show it", text: "a mark seen in one walk only, uncapped, went to a second look beside the other walk's crop of the same panel with the same spot outlined (mapped through the panel frame); the look said same or unsure, so needs your eye; different keeps the label" },
  K17: { label: "the second look saw no damage in the box", text: "a mark seen in one walk only, uncapped, went to the two-question second look (lead decision 5 of 11/1); asked whether the outlined box on the mark's own crop holds a damage mark (a scratch, scuff, chip, dent or crack), it said no or unsure, so the mark is not confirmed and caps at needs your eye; only yes there and no at the other walk's spot keeps new or not seen at return" },
  K18: { label: "the other walk shows the same patch", text: "registration (11/2): the pickup and return crops of the panel were aligned from the panel outline and its rigid edges, and at this mark's mapped spot the other walk's crop shows the same patch (normalized cross-correlation and the same contrast against its surroundings) with no mark of the other walk there, so the mark is not new or gone; needs your eye" },
  K19: { label: "the other walk boxed a mark over this spot", text: "a mark seen in one walk only whose box, carried into the other walk's panel frame, lies at least half inside a box Muse Spark drew in the other walk's compared view (often one large box such as a crush, a scuff area, dirt or a reflection); the other walk did not show clean paint there, so the mark is not called new or gone; needs your eye (thresholds matching.inside_other_box, 11/2)" },
  K20: { label: "the box does not stand out", text: "a mark seen in one walk only whose box does not stand out from its own surroundings in plain pixels: on the high-passed colour image of its own crop, the box's brightest tenth sits under min_contrast ring spreads above the ring around it (the registration presence test's contrast measure). Muse Spark may have boxed clean paint, so the mark is not called new or gone; needs your eye. A box over max_panel_share of its panel is an area such as a crush and is not tested (thresholds matching.stands_out, 11/2)" },
  L1: { label: "not seen at return", text: "d is None for a pickup mark: not_seen_at_return" },
  L2: { label: "new or unseen", text: "d is None for a return mark: new_or_unseen" },
  L3: { label: "already there", text: "d < tau_pre_r" },
  L4: { label: "needs your eye", text: "tau_pre_r <= d < tau_eye_r, or any K flag lowered the label" },
  L5: { label: "beyond the band", text: "d >= tau_eye_r: both marks keep their unmatched labels" },
  L6: { label: "not comparable", text: "a G rule failed for the panel; both best frames still show" },
  L7: { label: "outline changed in place", text: "G4 alone failed on every keyframe pair tried, the best whitened mask IoU is from 0.5 to under 0.7, the two keyframes' box aspect ratios agree within 2% and the largest blob of the mask difference holds at least half of it: the panel may be reshaped, so needs your eye; never already there or new" },
  E1: { label: "no car panel", text: "the claim image shows no car panel" },
  E2: { label: "naming disagreed", text: "Spark and the SAM vote name different panels: two candidates and one tap, recorded as the renter's input" },
  E2b: { label: "side unknown", text: "no side from the text or Spark: both sides shown" },
  E3: { label: "never filmed", text: "the named panel was not filmed at pickup" },
  E4: { label: "only glimpsed", text: "the named panel was glimpsed, not filmed: the sharpest frame and the seconds show" },
  E5: { label: "cannot place", text: "the mask touches the border and the landmark fit is off or has under 3 landmarks" },
  E6: { label: "no marks in ROI", text: "marks_v1 found nothing in the padded ROI" },
  E7: { label: "drawn box is a region", text: "the drawn box covers over 40% of the panel" },
  E8: { label: "claim crop condition", text: "glare, wet or dirt on the claim crop: capped at needs your eye" },
};

export function ruleTitle(id: string): string {
  const r = RULES[id];
  return r ? `${id} ${r.label}: ${r.text}` : id;
}

export function ruleLabel(id: string): string {
  return RULES[id]?.label ?? id;
}
