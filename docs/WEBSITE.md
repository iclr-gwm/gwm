# GWM project website

This folder contains the static project website for the anonymous GWM release.

## Files

- `index.html` - page structure and tabbed sections at the GitHub Pages publishing root.
- `assets/css/styles.css` - responsive visual system inspired by the paper figures.
- `assets/js/site-data.js` - website data for figures, tables, architecture, and the number index.
- `assets/js/app.js` - tab navigation and interactive renderers.
- `assets/favicon.svg` - site favicon.
- `website/index.html` - tiny redirect for older `/website/` links.
- `.nojekyll` - GitHub Pages marker for serving the static files directly.

## Source references

The website content is drawn from:

- the anonymous draft paper supplied as an attachment for this build;
- `../README.md`;
- `../IMPLEMENTATION.md`;
- `../examples.md`;
- `three_mode_live.md`.

The site intentionally does not include author names, affiliations, private case-study details,
local filesystem paths, or raw screenshots containing benchmark-domain labels. Paper figures are
represented as sanitized schematics, and paper tables are transcribed only at the level needed for
public project navigation.

Open `index.html` directly in a browser to view the site.

## Landing-page animation

The landing page embeds `assets/animation/index.html`, a self-contained copy of the
Figure 1 walkthrough. Its original source is preserved outside this repository.
The model API label is provider-neutral. Playback, scenario selection, timeline,
phase navigation, expanded view, keyboard shortcuts, and reduced-motion behavior
are retained. `assets/gwm-figure-1-animation.zip` contains the same copied files.

The website uses a local Helvetica/Arial sans-serif stack, consistent type sizes,
and responsive navigation. No remote font service or build step is required.
Preview the publishing root with `python3 -m http.server 8765 --directory docs`.

## Research-page visuals

`assets/css/research.css` styles the Paper, Results, Extension, and Resources views.
The Paper figures are responsive explanatory diagrams, not facsimiles of the PDF.
Figure 2 and the Results transfer chart share the actual Table 7 data: paired
point marks on a common 35–55% detail scale, with correctly positioned baselines,
direct values, distinct shapes, and unavailable entries preserved.
Table 2 and the primary results include reported sample SDs across repeats.
Native HTML diagrams keep their labels readable without scaling raster images.
The runtime diagram distinguishes setup, request handling, and future updates.

## Table signals and PDF references

Every Paper table has an evidence summary, selected cell emphasis, and scope notes.
`tableSignals` stores the claims, exact printed PDF page/line ranges, and zero-based
cell coordinates separately from the transcribed numerical rows. Green emphasis
identifies evidence for the summary; it does not assert statistical significance.
Table 4 highlights construction inventory, not performance or scalability.

References were checked against the 44-page numbered anonymous manuscript
`gwm-iclr2027-first-draft.pdf`, SHA-256 `b6880028579cdd15b753b27a365e9977130a8244878c937e603b412894c62abe`.
Recheck the printed page and margin-line references if the manuscript is revised.
The source PDF is not duplicated into the public website.

## vLLM + GWM extension animation

`assets/extension-animation/index.html` is a standalone, responsive walkthrough
embedded at the top of the Extension tab. It follows adapter preparation, one
agent request with K = 2, two internal candidate continuations, graph-grounded
selection, and one selected response. The agent executes any returned tool call.
An Agent view hides the explanatory internals behind the single service surface.
The workflow is illustrative; adapter building precedes requests, and the graph
and acting model weights remain fixed during each shown request.

The HTTP request example uses flat `gwm.*` keys in `vllm_xargs`, matching the
serving protocol. Playback supports direct phase navigation, pause/replay, a
timeline, and reduced motion, and suspends when the walkthrough is out of view.


## Compact explorer layouts

- Research tabs use tighter headings, selectors, figure spacing, and table rows while preserving the source data and evidence notes.
- The extension component inspector places the selected explanation beside the controls on wider screens and immediately below them on phones. The full interactive architecture map is an optional disclosure.
- User selections bring the local figure, table, component, or transfer explorer into view only when needed, accounting for the sticky header and reduced-motion preferences.
- At intermediate widths, the method schematic uses adjacent discovery/runtime lanes, and paired transfer plots remain side by side when their labels fit.
- Install and Serve with adapters use dark terminal cards. Commands wrap visually on narrow screens; Copy retains the original source text.


## Extension quickstart and project license

- The extension service boundary uses a pale green fill in the walkthrough.
- The hero review badge reads “Under double-blind review by ICLR 2027”; the brand subtitle is removed.
- Featured terminal cards show one install command, a chained build-and-serve command, and a single chat-completion request. Model, preset, input corpus, and runtime setup requirements are explicit.
- Downloadable request and rollout-format examples live in `assets/examples/`. The request demonstrates a text continuation from prior tool-call history; the rollout is a schema example, not a sufficient graph-discovery corpus.
- The project license, package metadata, README/website badges, and existing source SPDX headers use MIT. Anonymous project attribution and benchmark provenance descriptions are retained.

## Figure 01 entry and request JSON

- “See GWM in action” positions Figure 01 immediately below the sticky navigation,
  restores compact view, and starts at 00:00 on every click, including repeated
  clicks at the same hash. A frame-load fallback handles early clicks.
- The extension quickstart includes a visible, formatted Request xargs JSON card
  with Copy. Its settings match the downloadable multi-turn request example.
- The repository README highlights the project website with a badge and a callout.


## Related work and comparison

The Related work tab adapts the current manuscript’s `sections/related-work.tex`,
`sections/appendix-related-work.tex`, `tables/related-methods-summary.tex`, and
`references.bib`. It retains all 20 method rows, all 8 benchmark rows, and the
41 numbered references from the extended comparison. Paper titles link directly
to primary sources. Only active manuscript content is used; source comments and
private provenance are excluded.

The method table keeps the original six dimensions. GWM appears first in every
filter; other rows keep manuscript order within four semantic categories. Memory
and reuse is the default view. All rows remain available in the static HTML.
Benchmark context is separate from method design and identifies only CRMArena-Pro
and EnterpriseOps-Gym as used in this study.

These are qualitative design comparisons, not measured superiority. The section
retains the distinction between adapted evidence-format controls and complete
method reproductions, the frozen LLM in the prior Graph World Model’s embedding
variant, and the frozen backbone in Latent Action Reparameterization. No new
experimental results or manuscript PDFs are introduced.

`assets/css/related-work.css` and `assets/js/related-work.js` provide the responsive
layout and accessible filtering. Tables have horizontal scrolling on narrow
screens, row/column headers, visible focus, and a live filter count.

Source snapshots (SHA-256, September 28, 2026):

- `sections/related-work.tex`: `3a4b1cef2c1182b754f71aa19f411990467a6b9ba417c602526a26137e88a17b`
- `sections/appendix-related-work.tex`: `2fbd236ceeb1dc8a761d7f45498f801721f90a497354b03b94caeb3f193034ba`
- `tables/related-methods-summary.tex`: `b82554e22222d6d755fe483ed6b57973a04db65306ae861cbca3e331b382b4ba`
- `references.bib`: `fd41f88ca461ad6435e830e8b1a1bdb4309388513e131209a9a0386fabb923a9`
