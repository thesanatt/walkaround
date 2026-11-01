# Walkaround web: design system and layout rules

Decided Nov 1, 2026, during the UI rebuild. This file says which design system the app uses, why, and the rules each
screen follows.

## The choice: Carbon, through `@carbon/styles` only

Walkaround is an evidence tool. The page has to look like a careful instrument, not a landing page. Carbon
(IBM, Apache-2.0) fits that: square corners, a strict type scale, a neutral gray ramp, real contrast tokens for light
and dark, and IBM Plex, which the app already shipped through `@fontsource`.

What the app installs: `@carbon/styles` 1.118.0 (Sass) and `sass` 1.107.1 as a dev dependency. `src/carbon.scss`
loads only the theme tokens (`white` for light, `g100` for dark), the layout size tokens the button reads, and three
component styles: button, tag and progress bar. `src/styles.css` maps the app's own token names (`--bg`, `--ink`,
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
  - Home: "Start with your two videos". Secondary: "Open the sample pair".
  - Upload: no button until both videos are sealed; then "Open the result".
  - Result: "Open the sealed evidence sheet". Secondary: "Place a claim photo".
  - Sheet: "Download the PDF".
  - Claim: "Place the claim against the pickup walk".
- The result page leads with the answer in large plain words, counted from the verdicts code wrote
  (`src/lib/result.ts`): "1 mark was already there at pickup. 0 new. 11 need your eye." Then one card per mark, grouped
  in the order of that sentence: pickup frame and return frame side by side, zoomed around the mark, the label and one
  short line of why. Models, hashes, timestamps, the ledger, rule ids, both walk videos, coverage, Stays Put, narration
  and privacy are all inside "How it was checked", a `<details>` closed by default and complete when open.
- Every frame from licensed footage carries the run's `footage_credit` below it (CC BY 4.0 for the uCO3D pair).
- Verdict colours are Carbon's 60 grades on white and 40 grades on g100: green for already there, yellow for needs
  your eye, red for new, cool gray for not seen at return. Each label is also a word, never colour alone.
- Type: IBM Plex Sans 16 px body, line height 1.5; the answer at 42 px (32 px on a phone), weight 400. Lines of running
  text stay under about 64 characters.
- The grid breaks once on purpose: the two walk videos inside "How it was checked" take the extra width of a wide
  screen. The home picture starts lower than the words, so the page does not read as a centred banner.

## What the no-AI-slop rules mean here

No gradients, glass, glows, indigo or violet; no centred hero with three icon cards and an FAQ; no fade-up on scroll;
no count-up numbers; no emoji icons. Motion is limited to the progress fill and Carbon's button colour fade, and both
stop under `prefers-reduced-motion`. The kiosk keeps its own timed steps and its own reduced-motion rules.

## Where things live

- `src/carbon.scss`: the Carbon layer.
- `src/styles.css`: tokens mapped to Carbon, base type, then one section per screen and component.
- `src/lib/result.ts`: the answer sentence, the not-comparable line and the why line per rule.
- `src/components/MarkCard.tsx`: one mark, both walks, zoomed.
- `src/pages/HomePage.tsx` and `src/pages/UploadPage.tsx`: the two first screens (`/` and `/upload`).
