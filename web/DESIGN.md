# Walkaround web: design and layout rules

Design D, Nov 4, 2026, replacing the Carbon look of Nov 1 (samples A, B and C were turned down as too busy). Minimal,
little text, one easy path: start, answer, one finding at a time. The face is the video title's.

The product is called Alibi. Wherever the app shows its name it draws the wordmark ALIBI. (capitals, with the full
stop) in Titillium Web Black: the top row of every screen and the kiosk. The page title is Alibi and running text says
Alibi. URLs, the package, the CLI and code identifiers keep walkaround. The privacy line still mirrors
`walkaround/privacy.py` word for word, and the sealed sheets name Walkaround, so those say Walkaround until the backend
is renamed.

## The look

- Pure black `#000`, one warm white `#f3f0ea` for text, one grey `#8f8f8a` for secondary text (6.4:1 on black), hairlines
  `#262626`. Square corners. No cards with shadows, no pills, no gradients, no glow, no icons.
- The page background carries a faint static fibre texture (`src/textures/fibre.png`, a 128 px tile of curved strands),
  off on phones and under `prefers-contrast` and `forced-colors`. Frames and thumbnails stay flat.
- Colour only where the renter must act: brake red `#ff3b2b` for new, amber `#ffbf1f` for needs your eye. Already there,
  gone at return and everything else stay white or grey. Every label is also a word.
- Type: Titillium Web (OFL, self-hosted in `src/fonts/`). Black only for the ALIBI. wordmark and the answer, Bold for
  the start headline, Regular for everything else; IBM Plex Mono (`@fontsource`) only for code and hashes in the
  math and the sealed panel. Body 18 px. The answer is Titillium Black at up to 80 px. Text that fits on two lines is
  centred; longer text on a phone is left-aligned.
- Motion: the progress line of an upload, the button colour fade and the sealed panel sliding in (160 ms). All three
  stop under `prefers-reduced-motion`.
- A screen at least 1800 x 1000 (the 1920 x 1080 demo recording) lays the app out at 1600 x 900 and scales it 1.2, so
  every screen keeps its 1440 x 900 composition at a size that reads on video.
- The kiosk keeps its own 1920 x 1080 stage and pins `data-theme='light'`; `src/styles.css` carries light values for
  it, and the shared components (labels, verdict cards, the dates card) follow D there too.

## The screens

1. Start (`/`, `/upload`): "Which scratches are new since you got the car?", one muted line on how to film, two empty
   16:9 frames side by side with the label under each (the same in-browser hashing and resumable upload as before),
   then Compare, the one filled shape, once both videos are in ("Add both videos" until then), See an example and the
   privacy line. On a replay-only copy the slots
   rest, one line says new checks are off, and the button opens the example. Options (pickup only, keep the plate) sit
   in a closed row.
2. Result (`/run/<id>`): the answer first and very large, from the verdicts code wrote: "1 new mark" in red, or "No
   new marks" in white. One line under it ("Right rear door. Not found in the pickup video." or "3 spots need your
   eye."), then the new mark's pickup and return frames side by side, the widest thing on the page, and Show the math.
   Then one row per finding with a thumbnail: New, Needs your eye, Already there (closed until opened), then Gone at
   return when there is one. Panels compared and wheels and glass get one quiet line. Panels not compared and the roof
   stay on the sealed sheet. The hero sample alone gets a caption under its frames saying what is tape and what is
   real, and the same caption on its finding pages.
3. Finding (`?f=<verdict_id>`): the label, the panel, Muse Spark's words and the why line, then pickup and return
   side by side (stacked on a phone), 3:2, zoomed on the mark. Every box is drawn in screen pixels, at least 56 px a
   side and padded. A video with no mark outlines the same spot with a black and white dash, from the run's registration (the found spot
   when the patch was found there). Left and right arrows step, Escape goes back. Show the math (`&math=1`) gives the
   deciding comparison large ("0.915 > 0.849"), then the cost, the second look and the rule, each with the real lines
   of `match.py` or `register.py` in VS Code Dark+ colours and the trace's values on the right.
4. Your check: a needs-your-eye finding takes one tap. A mark seen at pickup only: Gone at return or Still there; at
   return only: New or Already there; otherwise Already there or New. The finding then shows the answer as its label
   with a "your check" tag, the result counts it under that answer (tagged), and the sealed panel and the sheet page
   list it under "Your checks (not part of the sealed sheet)". Stored per run in localStorage, undoable. Code's labels
   and the sheet never change.
5. Sealed: a text link in the top row with the local time and zone opens the hashes, the RFC 3161 and Bitcoin stamps,
   the PDF, the sheet, verify and bundle.zip, then the renter's checks, then "Everything the check recorded".
6. Everything the check recorded (`?record=1`): one page of sections, each a heading, one grey line and the facts:
   the two walks with SAM 3.1's outlines and their seals, what each model did, coverage, panels, crops, the match,
   every verdict with its rule, the panels not compared and the lines left to check by eye, Stays Put (every tracked
   mark folded), narration, the claim, the sheet files and privacy, then the run itself. This is the only screen that
   lists the not-compared panels and the roof. A running check shows the same page with the walks first and the
   sections folded under "How it is being checked"; a pickup-only check shows its marks at pickup; a failed one opens
   every section.
7. A claim photo placed on a run (`?claim=1`, a row at the top of the result): the bill's photo beside the sealed
   pickup frame of that panel, whole, the bill's words and one line of where it landed, then "Show how it was placed".
   `/claim` takes a new one; a replay-only copy says new claims are off and links the finished one.
8. No run at the address: "No check at this address", what the server said, Start over and See an example.

Show the math opens with one line of who did what (SAM 3.1, Muse Spark, Muse Spark Open, and that code made the call).

A replay says so beside the name in the top row on every screen of it. Every frame from licensed footage carries the
run's `footage_credit` under it.

## What the no-AI-slop rules mean here

No gradients, glass, glows, indigo or violet; no centred hero with icon cards; no fade-up on scroll; no count-up
numbers; no emoji; no caps-tracked eyebrows. The one break in the grid is the result's two columns, the answer held on
the left.

## Where things live

- `src/styles.css`: the tokens, the fonts, base type and the components the record view, the claim page and the
  kiosk share (seal, coverage car, panels, crops, match, verdicts, Stays Put, narration, claim).
- `src/screens.css`: the start, result, finding, math, sealed, record, claim and not-found screens.
- `src/lib/findings.ts`: the answer lines, the list and its groups, the frames of a finding and the answers a check
  offers. `src/lib/yourcheck.ts`: the checks in localStorage.
- `src/lib/math.ts`, `src/lib/mathlines.ts` and `src/lib/pycode.ts`: the math panel's steps, the quoted lines
  (`test/design_d.test.ts` checks each against the file) and the Dark+ colouring.
- `src/components/Frame.tsx`: one walk's zoomed frame with the boxes, and `zoomWindow`. `SealedPanel.tsx`,
  `MathPanel.tsx`, `Slot.tsx`, `Bar.tsx` (with the wordmark and the replay note), `YourChecks.tsx`.
- `src/pages/StartPage.tsx`, `src/pages/RunPage.tsx` (result, finding, claim, sealed, not found),
  `src/pages/RunRecord.tsx` (everything recorded, and a running check) and `src/pages/ClaimPage.tsx`.
