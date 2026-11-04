# Third-party licenses

The browser build committed in `web/dist/` bundles the code and fonts below. Each license text is copied unchanged
from the package's own LICENSE file in `web/node_modules/`, or for Titillium Web from google/fonts. The list comes from
the build's source maps (JavaScript) and from what `web/src/main.tsx` and `web/src/styles.css` pull into the CSS
(fonts).

| Package | Version | License | Text |
|---|---|---|---|
| @meta-sam/parser, @meta-sam/react, @meta-sam/video, @meta-sam/graphics | 0.1.0, 0.1.14, 0.1.5, 0.1.13 | SAM License | `SAM_LICENSE.txt` |
| react, react-dom, scheduler | 19.3.1, 19.3.1, 0.28.0 | MIT | `react-MIT.txt` |
| hash-wasm | 4.12.0 | MIT | `hash-wasm-MIT.txt` |
| mediabunny (pulled in by @meta-sam/video) | 1.55.7 | MPL-2.0 | `mediabunny-MPL-2.0.txt` |
| IBM Plex Mono font, via @fontsource | 5.3.0 | SIL Open Font License 1.1 | `ibm-plex-mono-OFL-1.1.txt` |
| Titillium Web font (Regular, SemiBold, Bold, Black), from github.com/google/fonts `ofl/titilliumweb`, converted to WOFF2 with fontTools, glyphs unchanged, in `web/src/fonts/` | Google Fonts release | SIL Open Font License 1.1 | `titillium-web-OFL-1.1.txt` |

mediabunny is bundled unmodified. Its source is at https://github.com/Vanilagy/mediabunny and on npm as
`mediabunny@1.55.7`.

The Python side installs `meta-sam-parser` from PyPI at install time; it is under the same SAM License and is not
copied into this repo.

Everything else in this repo is MIT (`../LICENSE`), except the footage, which `../README.md` credits under License.
