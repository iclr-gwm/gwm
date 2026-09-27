# GWM Figure 1 animation

An interactive walkthrough of Graph World Model (GWM) construction and use. Two scenarios show how workflow evidence can guide an agent after an initial task failure: **Enterprise Report Access** and **Banking Report Approval**, selected under **Example**.

## Open and use

Open `index.html` in a modern browser. The page is self-contained: its styles, scripts, icons, and diagrams are embedded. No installation, package manager, build step, internet connection, or external resource is required.

The default view is a compact figure for embedding within a larger page. It shows the active phase in a stable frame, with a slim model connection row, current-action highlight, and playback controls. **Expand figure** opens the full figure; switching views preserves playback position, model selection, and scenario. In compact view, use the scenario menu and the three phase buttons to explore.

**Defaults:** autoplay on, loop on, served model via vLLM Extension. A reduced-motion preference starts the animation paused. The Loop checkbox, model menu, play/pause, replay, scrubber, and speed control remain available.

At the start of **Step 3 · With GWM**, the completed Step 2 graph is reused in the same compact graph area with a short, gentle handoff. There is no flying overlay or position measurement during the transition. In expanded view the original graph remains in Step 2, while the copy appears in Step 3. Graph structure and scenario labels are preserved; only retrieved evidence is highlighted during use.

The graph shows a single relevant edge label at a time. Labels use concise call names such as `create_permission(…)`, `request_approval()`, and `share_report(…)`; messages and observations remain distinct. Hover over an edge to reveal its label and full description. Details and arguments are retained in accessible descriptions instead of crowding the diagram. These names are illustrative, not an API specification. `request_approval()` only requests approval; the separate reviewer response supplies it.

The visual sequence includes animated tool routes, blocked and restored report access, moving history embeddings, cluster-to-graph construction, retrieved evidence, and candidate selection.

A yellow highlight follows the current action. The **Now** label names that action; red and green remain reserved for failure and success. The highlight follows playback and timeline scrubbing, including when paused.

The model diagram offers **Model API** and **Served via vLLM Extension** views. It separates the model, vLLM Extension, agent loop, and tools, and indicates when the model generates a response or candidates and when the agent runs a tool. The served view connects through the vLLM Extension; the API view connects directly. These are illustrative access routes, not live connections. Switching the model view preserves the playback position. The public technology labels follow the requested wording. No real endpoints, credentials, account identifiers, or deployed-model identifiers are included.

Each scenario runs for approximately 36 seconds at normal speed, then loops by default. In expanded view, selecting a numbered step pauses on that step's completed state. Compact view shows one phase at a time at all widths; **Expand figure** reveals the full layout. Keyboard users can tab through the controls. When focus is outside a control, Space toggles playback and the left/right arrow keys move between steps.

Keep `index.html` and this README together in this folder. The HTML file is the complete page; there are no additional assets to fetch.

## Embed within another page

The compact desktop figure is approximately **960 × 600 pixels**. It fills narrower containers responsively. A phone layout is taller to keep graph labels readable. The following example reserves room for each layout and uses only a relative link:

```html
<style>
  .gwm-animation {
    display: block;
    width: 100%;
    max-width: 960px;
    height: 620px;
    margin: 0 auto;
    border: 0;
  }
  @media (max-width: 760px) {
    .gwm-animation { height: 680px; }
  }
  @media (max-width: 540px) {
    .gwm-animation { height: 860px; }
  }
</style>
<iframe
  class="gwm-animation"
  src="./gwm-figure-1-animation/index.html"
  title="Graph World Model: construction and guided use"
  loading="lazy">
</iframe>
```

Adjust the relative `src` to where the folder is placed. The optional expanded view can scroll inside a fixed-height iframe. No external fonts, libraries, or asset URLs are needed.

## Publish with GitHub Pages

1. Copy this folder into the branch and directory configured as the site's publishing source.
2. If publishing from a branch, enable Pages for that branch and publishing directory in the repository settings. If a Pages deployment workflow already exists, include this folder in its published output.
3. Open the site's address followed by `/gwm-figure-1-animation/`. If the folder contents are placed directly at the publishing root, open the site's root address instead.

All resources travel with `index.html`, so the page works under a repository subpath without asset-path configuration. Publication is a separate deployment action; opening the file locally does not publish it.

## Privacy

The walkthrough uses synthetic wording and generic roles, with the user-selected public technology labels vLLM Extension and Model API. It includes no real personal or customer-organization identities. It contains no source screenshots, raw interaction logs, private source links, account identifiers, author metadata, or local filesystem paths. Historical material informed the adaptation but is not included in this folder.

The page requires no credentials, remote services, analytics, or externally hosted fonts or libraries. Playback stays in the browser.

## Scientific scope

The animation explains the Figure 1 mechanism; its guided conversations and displayed candidate choices are schematic. It does not replay a recovered historical decision trace or present a measured causal comparison.

- **Initial attempt:** An agent leaves a prerequisite unresolved. In report access, access is still missing after an error. In report approval, sharing is blocked because approval has not been received.
- **Offline construction:** Multiple successful and failed discovery rollouts provide separate prior experience. The displayed task is not the discovery corpus. Interaction-history prefixes are embedded and clustered into workflow states; directed edges record observed successor relations. The memory stores workflow evidence such as toolsets, outcome associations, support, and examples.
- **Guided use:** Retrieved evidence supplies advice before candidate generation. Selection chooses among the agent's proposed continuations before tool execution. The agent completes the missing prerequisite and observes the result.
- **Fixed during use:** Actual tool observations update live interaction history. The workflow graph and agent weights remain fixed; the animation does not depict online graph updates or policy training.
- **Separate domains:** Each scenario uses its own discovery material and workflow memory. The animation does not claim transfer between the scenarios. Approval in the approval scenario comes from a reviewer, not from GWM.

GWM refers to the combined workflow-memory guidance method, including retrieval, language-model guidance, and the software procedure that supplies advice or selection. The stored representation is the workflow graph or memory. Its states and edges summarize observed interaction patterns; they do not establish universal rules, guaranteed outcomes, or isolated causation by the graph.

## Compact website version

The copied website version uses a shorter compact stage and explicit phase buttons.
“Click a step to jump” identifies the three interactive phases. The parent website
sizes its embedded frame to the compact content height; expanded view remains
available through the control or the standalone animation link.

## Landing-page playback

The parent page can send `{ "type": "gwm:restart" }` to restart at 00:00,
restore compact view, and play. The embedded figure accepts messages only from
its parent and replies with `gwm:restarted` and its rendered height. The parent
aligns Figure 01 below the sticky navigation and handles clicks before the frame
has finished loading. `gwm:visibility` messages suspend progress off screen.
Explicit playback keeps reduced-motion effects disabled when requested.
