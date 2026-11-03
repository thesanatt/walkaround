# Third-party licenses

The browser build committed in `web/dist/` bundles the code, styles and fonts below. Each license text is copied
unchanged from the package's own LICENSE file in `web/node_modules/`. The list comes from the build's source maps
(JavaScript) and from what `web/src/carbon.scss` and `web/src/main.tsx` pull into the CSS (styles and fonts).

| Package | Version | License | Text |
|---|---|---|---|
| @meta-sam/parser, @meta-sam/react, @meta-sam/video, @meta-sam/graphics | 0.1.0, 0.1.14, 0.1.5, 0.1.13 | SAM License | `SAM_LICENSE.txt` |
| react, react-dom, scheduler | 19.3.1, 19.3.1, 0.28.0 | MIT | `react-MIT.txt` |
| hash-wasm | 4.12.0 | MIT | `hash-wasm-MIT.txt` |
| mediabunny (pulled in by @meta-sam/video) | 1.55.7 | MPL-2.0 | `mediabunny-MPL-2.0.txt` |
| @carbon/styles (compiled into the CSS) | 1.118.0 | Apache-2.0 | `carbon-Apache-2.0.txt` |
| IBM Plex Sans and IBM Plex Mono fonts, via @fontsource | 5.3.0 | SIL Open Font License 1.1 | `ibm-plex-sans-OFL-1.1.txt`, `ibm-plex-mono-OFL-1.1.txt` |

mediabunny is bundled unmodified. Its source is at https://github.com/Vanilagy/mediabunny and on npm as
`mediabunny@1.55.7`.

The Python side installs `meta-sam-parser` from PyPI at install time; it is under the same SAM License and is not
copied into this repo.

Everything else in this repo is MIT (`../LICENSE`), except the footage, which `../README.md` credits under License.
