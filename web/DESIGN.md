# Walkaround web: design and layout rules

Design D, Nov 4, 2026, replacing the Carbon look of Nov 1 (samples A, B and C were turned down as too busy). Minimal,
little text, one easy path: start, answer, one finding at a time. The face is the video title's.

The product is called Alibi. Wherever the app shows its name it draws the wordmark ALIBI. (capitals, with the full
stop) in Titillium Web Black: the top row of every screen and the kiosk. The page title is Alibi and running text says
Alibi. URLs, the package, the CLI and code identifiers keep walkaround. The privacy line still mirrors
`walkaround/privacy.py` word for word, and the sealed sheets name Walkaround, so those say Walkaround until the backend
is renamed.

## The look

- Pure black `#000`, one white `#f4f4f1` for text, one grey `#8f8f8a` for secondary text (6.4:1 on black), hairlines
  `#262626`. Square corners. No cards with shadows, no pills, no gradients, no glow, no icons.
- Colour only where the renter must act: brake red `#ff3b2b` for new, amber `#ffbf1f` for needs your eye. Already there,
  gone at return and everything else stay white or grey. Every label is also a word.
- Type: Titillium Web (OFL, self-hosted in `src/fonts/`, Regular, SemiBold, Bold, Black) for words; IBM Plex Mono
  (`@fontsource`) for code, hashes, times and the numbers in the math. Body 18 px. The answer is Titillium Black at
  up to 104 px.
- Motion: the progress line of an upload, the button colour fade and the sealed panel sliding in (160 ms). All three
  stop under `prefers-reduced-motion`.
- The kiosk keeps its own stage and pins `data-theme='light'`; `src/styles.css` carries light values for it.

## The screens

1. Start (`/`, `/upload`): "What changed between pickup and return?", the two video slots (the same in-browser hashing
   and resumable upload as before), one button, Compare, and a quiet See an example. On a replay-only copy the slots
   rest, one line says new checks are off, and the button opens the example. Options (pickup only, keep the plate) sit
   in a closed row.
2. Result (`/run/<id>`): the answer first and very large, from the verdicts code wrote: "1 new mark" and the panel, or
   "No new marks" and how many need your eye. Then one row per finding with a thumbnail: New, Needs your eye, Already
   there, then Gone at return when there is one. Wheels and glass get one quiet line. Panels not compared and the roof
   stay on the sealed sheet. On a wide screen the answer stays on the left while the list scrolls.
3. Finding (`?f=<verdict_id>`): the label, the panel, Muse Spark's words and the why line, then pickup and return
   side by side (stacked on a phone), 3:2, zoomed on the mark. Every box is drawn in screen pixels, at least 56 px a
   side and padded. A walk with no mark outlines the same spot, dashed, from the run's registration (the found spot
   when the patch was found there). Left and right arrows step, Escape goes back. Show the math (`&math=1`) gives the
   deciding comparison large ("0.915 > 0.849"), then the cost, the second look and the rule, each with the real lines
   of `match.py` or `register.py` in VS Code Dark+ colours and the trace's values on the right.
4. Your check: a needs-your-eye finding takes one tap. A mark seen at pickup only: Gone at return or Still there; at
   return only: New or Already there; otherwise Already there or New. The finding then shows the answer as its label
   with a "your check" tag, the result counts it under that answer (tagged), and the sealed panel and the sheet page
   list it under "Your checks (not part of the sealed sheet)". Stored per run in localStorage, undoable. Code's labels
   and the sheet never change.
5. Sealed: a text link in the top row with the local time and zone opens the hashes, the RFC 3161 and Bitcoin stamps,
   the PDF, the sheet, verify and bundle.zip, then the renter's checks, then "Everything the check recorded"
   (`?record=1`, the full record of the run: walks with SAM 3.1's outlines, each model's part, coverage, panels, crops,
   the match, verdicts, Stays Put, narration, privacy). A running, pickup-only or failed check shows that record view.

A replay says so beside the name in the top row on every screen of it. Every frame from licensed footage carries the
run's `footage_credit` under it.

## What the no-AI-slop rules mean here

No gradients, glass, glows, indigo or violet; no centred hero with icon cards; no fade-up on scroll; no count-up
numbers; no emoji; no caps-tracked eyebrows. The one break in the grid is the result's two columns, the answer held on
the left.

## Where things live

- `src/styles.css`: the tokens, the fonts, base type and the older components (record view, claim, kiosk).
- `src/screens.css`: the start, result, finding, math and sealed screens.
- `src/lib/findings.ts`: the answer lines, the list and its groups, the frames of a finding and the answers a check
  offers. `src/lib/yourcheck.ts`: the checks in localStorage.
- `src/lib/math.ts`, `src/lib/mathlines.ts` and `src/lib/pycode.ts`: the math panel's steps, the quoted lines
  (`test/design_d.test.ts` checks each against the file) and the Dark+ colouring.
- `src/components/Frame.tsx`: one walk's zoomed frame with the boxes. `SealedPanel.tsx`, `MathPanel.tsx`, `Slot.tsx`,
  `Bar.tsx`, `YourChecks.tsx`.
- `src/pages/StartPage.tsx`, `src/pages/RunPage.tsx` (result, finding, sealed) and `src/pages/RunRecord.tsx`.
