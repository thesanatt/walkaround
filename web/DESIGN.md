# Walkaround web: design system and layout rules

Decided Nov 1, 2026, during the UI rebuild. This file says which design system the app uses, why, and the rules each
screen follows.

## The choice: Carbon, through `@carbon/styles` only

Walkaround is an evidence tool. The page has to look like a careful instrument, not a landing page. Carbon
(IBM, Apache-2.0) fits that: square corners, a strict type scale, a neutral gray ramp, real contrast tokens for light
and dark, and IBM Plex, which the app already shipped through `@fontsource`.

What the app installs: `@carbon/styles` 1.118.0 (Sass) and `sass` 1.107.1 as a dev dependency. `src/carbon.scss`
loads only the theme tokens (`g100` dark on every page whatever the system prefers, `white` where a page pins
`data-theme='light'`, as the kiosk does unless `?theme=dark`), the layout size tokens the button reads, and three
component styles: button, tag and progress bar. The evidence sheet and its PDF stay light. `src/styles.css` maps the app's own token names (`--bg`, `--ink`,
`--line`, `--accent` and the rest) onto Carbon's `--cds-*` tokens, so every older component and the kiosk follow
Carbon without renaming a class. That matters because the tests assert component markup and class names.

### Compared

| system | license | for this codebase | against it here |
|---|---|---|---|
| Carbon (`@carbon/react`, `@carbon/styles`) | Apache-2.0 | IBM Plex is already the app's font; tokens for light and dark; reads as a serious tool | `@carbon/react` is heavy for what we use (below) |
| Primer (`@primer/react`) | MIT | solid, accessible | reads as GitHub; brings its own theme provider and styling system next to our CSS |
| Radix Themes (`@radix-ui/themes`) | MIT | good primitives, small | default accent is indigo and the default look (rounded cards, soft shadows) is the generic AI-template look the no-slop rules ban; restyling it fully is more work than it saves |
| Mantine (`@mantine/core`) | MIT | many components | default look is the same rounded, shadowed template; needs its provider and PostCSS preset; most of its components we would not use |

### Why not `@carbon/react`

Measured on Nov 1 with esbuild, minified and gzipped, against an empty React page: importing only `Button` and
`ProgressBar` from `@carbon/react` 1.119.0 added 20.4 kB gzipped (69.3 to 89.7 kB). `Button` pulls `IconButton`,
`Tooltip`, `Popover` and floating-ui even when no icon is used. The markup of a Carbon button, tag or progress bar is a
few classes on a native element, so the app writes those classes itself (`cds--btn cds--btn--primary` on our router's
`Link`, which keeps in-app navigation). Collapsible sections are native `<details>`, which needs no script and keeps the
content in the page for find-in-page.

Also measured: Carbon's notification, radio and checkbox styles compile to 110, 24 and 26 kB of CSS. Those stay native
elements styled with the same tokens. The shipped CSS is 23 kB gzipped (it was 7 kB before Carbon).

### Install note

`@carbon/styles` depends on `@ibm/telemetry-js`, which reports package usage to IBM from a postinstall script unless
`IBM_TELEMETRY_DISABLED=true` is set. Install with:

```
IBM_TELEMETRY_DISABLED=true npm install
```

Nothing from it reaches the built app; it runs only at install time.

## Layout rules

- One primary action per screen, drawn as Carbon's primary button. Everything else is a tertiary button or a link.
  - Home: "See a finished check" while live checks are off (the first sample in `samples/index.json`); "Check your two
    videos" when they are on. The other one is a text link. The front door never says live checks are off, and the
    primary never leads to a dead end.
  - Upload: no button until both videos are sealed; then "Open the result".
  - Result: "Download the sealed PDF". Secondary: "A bill arrived? Place its photo" on a run that takes claims, else
    "Read the sheet here".
  - Sheet: "Download the PDF".
  - Claim: "Place the claim against the pickup walk", with a one-click link to a finished claim when a sample holds one.
- Home (second pass, Nov 1): the problem line, "Your pickup video, read like evidence.", one sentence, the action, and
  on the right the first answer card of the first sample (pickup and return side by side, the label). Then "Before you
  drive off" and "When a bill arrives", then "Finished checks you can open", one row per sample with a thumbnail
  cropped close to the mark (a claim sample shows its claim photo) and its answer line, then one line on the models.
- Result (second pass, Nov 1), top to bottom, all display over the labels code wrote (`src/lib/result.ts`):
  1. "Both walks sealed <time>, before any model saw them." The word sealed links to the verify page.
  2. The answer in the renter's terms, counted in panels: "1 new mark: left front door." or "Nothing new on the 3
     panels both walks show clearly." When any spot waits for the renter it says "No new damage confirmed" instead.
     Under it: there at pickup, spots that need your eye, panels not compared.
  3. A claim photo placed against the run, when there is one, beside the sealed pickup frame; the steps and the sheet
     paragraph sit in a closed row.
  4. The car map (`ResultMap.tsx`): compared panels outlined, not compared hatched, a numbered pin per answer card and a
     yellow pin per panel with spots to check; the list under it says each pin in words and jumps to its card. Beside
     it, `PanelClip.tsx` loops about three seconds of the walk with everything but the first card's panel dimmed
     (SAM 3.1's mask), paused with controls under reduced motion.
  5. The answer cards, new first, then already there: verdict word and panel as the heading, both frames zoomed around
     the mark, one line of why, and one quiet credit line naming which Meta model did what ("SAM 3.1 tracked this
     panel; Muse Spark found it in both walks; Muse Spark Open agreed it is the same spot; code made the call").
  6. Needs your eye as closed rows grouped by reason (`EYE_KINDS`), each with what the renter can do; open shows
     compact cards. Then not seen at return, then the panels not compared as one closed row with their reasons.
  7. "How it was checked", a `<details>` closed by default and complete when open: run id, timing, cost, the models
     strip, both walk videos with SAM's outlines, coverage, panels, crops, the match, verdicts with rule ids, Stays Put,
     narration, the sheet files and privacy. Rule ids, run ids and internal names appear only in there.
- Every frame from licensed footage carries the run's `footage_credit` below it (CC BY 4.0 for the uCO3D pair, NHTSA's
  terms for the crash test).
- Verdict colours are Carbon's 60 grades on white and 40 grades on g100: green for already there, yellow for needs
  your eye, red for new, cool gray for not seen at return. Each label is also a word, never colour alone.
- Type: IBM Plex Sans 16 px body, line height 1.5; the answer at 42 px (32 px on a phone), weight 400. Lines of running
  text stay under about 64 characters.
- The grid breaks once on purpose: the two walk videos inside "How it was checked" take the extra width of a wide
  screen. The home card starts lower than the words, so the page does not read as a centred banner.

## What the no-AI-slop rules mean here

No gradients, glass, glows, indigo or violet; no centred hero with three icon cards and an FAQ; no fade-up on scroll;
no count-up numbers; no emoji icons. Motion is limited to the progress fill and Carbon's button colour fade, and both
stop under `prefers-reduced-motion`. The kiosk keeps its own timed steps and its own reduced-motion rules.

## Where things live

- `src/carbon.scss`: the Carbon layer.
- `src/styles.css`: tokens mapped to Carbon, base type, then one section per screen and component.
- `src/lib/result.ts`: the answer line, the panel counts, the needs-your-eye groups, the credit line and the why line
  per rule.
- `src/components/MarkCard.tsx`: one mark, both walks, zoomed. `ResultMap.tsx`: the car map with pins.
  `PanelClip.tsx`: the SAM-masked loop of the first card's panel.
- `src/pages/HomePage.tsx` and `src/pages/UploadPage.tsx`: the two first screens (`/` and `/upload`).
