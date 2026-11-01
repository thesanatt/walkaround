# Claim on pairC_core3_claim2 (Nov 1, 2026, 06:40 UTC)

pairC_core3_claim2 is a non-kept copy of the sample run pairC_core3 (a kept run takes no claim), made with no unblurred
file. Its frozen thresholds are pairC_core3's plus the six matching leaves of the lead's second decision of Nov 1
(tight_pair_k5, reshaped_panel and the four reshape numbers), config_hash recomputed; the walk files are pairC_core3's
bytes. One claim through POST /api/claims (the app's code path): the photo cut from the return walk at 2.1 s, centred
on the trunk chips, plate and a bumper sticker pixelated (the same bytes as round 4's claim_4, sha256 83993d8d...), the
text "Two chips on the trunk lid by the edge, rear of the car" and the tap trunk_lid.

- Match: the claim mark (chip) paired with pickup_trunk_lid_kf1_m1 (dirt), d 0.734 inside tau_pre_r 0.769 (L3 by
  distance). K4 and K5 waited for the second look under tight_pair_k5.
- Second look, the provenance line (claim_1.open_weights.ndjson): model meta-models/muse-spark-open-1026, key_slot
  hackathon, engine baseten, HTTP 200, valid JSON, 6884 ms, request chatcmpl-de6971fb7574410c9364506329eac59a,
  verdict different. Its reason: "The box in the second image contains a distinct dark linear scratch/mark on the
  trunk lid that is absent in the corresponding clean white area outlined in the first image, showing the pairing..."
- Label: needs your eye, rule ids L4, L3, K7. The paragraph: "Muse Spark typed the pickup mark dirt and the claim mark
  chip (K4), two kinds that disagree (K5): a note here; the second look compared the two crops and said different
  (K7)."
- Cost (ledger_rows.jsonl): 19 rows, $0.0567 (16 SAM image checks $0.0400, 2 Muse Spark calls $0.0128, 1 Spark Open
  $0.0039 at Baseten list price, hackathon credits).
- claim.rebuild_claim with the copy's thresholds gives back claim_1.json byte for byte.
