(function () {
  const data = window.GWM_SITE_DATA;
  const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));

  function setTab(id, updateHash = true, moveFocus = false) {
    const target = panels.some((panel) => panel.id === id) ? id : "overview";
    tabButtons.forEach((button) => {
      const active = button.dataset.tab === target;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    panels.forEach((panel) => panel.classList.toggle("is-active", panel.id === target));
    syncWalkthroughVisibility();
    if (updateHash && location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
    if (moveFocus) {
      document.getElementById(target).focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "instant" });
    }
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      setTab(button.dataset.tab, true, true);
    });
  });
  document.querySelectorAll("[data-tab-jump]").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tabJump, true, true));
  });
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1) || "overview";
    if (id === "main-content") {
      document.getElementById("main-content").focus({ preventScroll: true });
      return;
    }
    setTab(id, false, panels.some((panel) => panel.id === id));
    if (id === "animation") alignFigureOne();
  });

  // Keep a user-selected explanation in view without moving an already visible explorer.
  function revealSelection(selector) {
    const section = document.querySelector(selector);
    if (!section) return;
    const top = document.querySelector(".site-header").getBoundingClientRect().bottom + 12;
    const box = section.getBoundingClientRect();
    const available = window.innerHeight - top - 16;
    const visible = Math.max(0, Math.min(box.bottom, window.innerHeight - 16) - Math.max(box.top, top));
    if (box.top < top - 1 || visible < Math.min(box.height, available) * 0.8) {
      window.scrollTo({
        top: Math.max(0, window.scrollY + box.top - top),
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth"
      });
    }
  }

  function renderFigureButtons() {
    const holder = document.getElementById("figure-buttons");
    holder.innerHTML = "";
    data.figures.forEach((figure, index) => {
      const button = document.createElement("button");
      button.type = "button";
      const labels = ["Construction & use", "Cross-benchmark reuse", "Rescues & regressions", "Memory & replay", "Actionable guidance", "Recovery paths"];
      button.innerHTML = `<span>${figure.label}</span><strong>${labels[index]}</strong>`;
      button.setAttribute("aria-label", `${figure.label}: ${figure.title}`);
      button.dataset.figure = figure.id;
      button.className = index === 0 ? "is-active" : "";
      button.setAttribute("aria-controls", "figure-art figure-title figure-summary");
      button.addEventListener("click", () => { renderFigure(figure.id); revealSelection(".paper-grid"); });
      holder.appendChild(button);
    });
    renderFigure(data.figures[0].id);
  }

  function renderFigure(id) {
    const figure = data.figures.find((item) => item.id === id) || data.figures[0];
    document.querySelectorAll("[data-figure]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.figure === figure.id);
      button.setAttribute("aria-pressed", String(button.dataset.figure === figure.id));
    });
    document.getElementById("figure-kicker").textContent = figure.label;
    document.getElementById("figure-title").textContent = figure.title;
    document.getElementById("figure-summary").textContent = figure.summary;
    const facts = document.getElementById("figure-facts");
    facts.innerHTML = "";
    figure.facts.forEach(([term, desc]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = desc;
      facts.append(dt, dd);
    });
    document.getElementById("figure-art").innerHTML = figureArt(figure.art);
  }

  function flowStep(number, title, description, focal = false) {
    return `<div class="method-step${focal ? ' focal' : ''}"><span class="step-index">${number}</span><strong>${title}</strong><p>${description}</p></div>`;
  }

  function transferPlot(target) {
    const spec = data.transfer[target];
    const name = target.toUpperCase();
    const source = target === "crm" ? "EOPS" : "CRM";
    const position = (value) => ((value - 35) / 20) * 100;
    function mark(label, value, kind) {
      return `<div class="dot-row"><span class="series-name">${label}</span><div class="dot-track" aria-hidden="true"><i class="baseline-mark" style="left:${position(spec.baseline)}%"></i>${value === null ? '' : `<i class="data-dot ${kind}" style="left:${position(value)}%"></i>`}</div><span class="dot-value">${value === null ? '—' : value.toFixed(1) + '%'}</span></div>`;
    }
    return `<section class="transfer-plot" aria-label="${name} target task success, graph from ${source}">
      <div class="plot-heading"><h4>${source} → ${name}</h4><span>Target baseline <strong>${spec.baseline.toFixed(1)}%</strong></span></div>
      <p class="plot-unit">Task success (%) · higher is better</p>
      <div class="plot-axis" aria-hidden="true"><span></span><div>${[35,40,45,50,55].map(v => `<span style="left:${position(v)}%">${v}</span>`).join('')}</div><span></span></div>
      ${spec.rows.map(([label, own, cross]) => `<div class="dot-group"><h5>${label.replace(' / ', ' discovery · ').replace('K=', 'K = ')}</h5>${mark('In-domain',own,'own')}${mark('Transferred',cross,'cross')}</div>`).join('')}
      <p class="plot-note"><span class="baseline-key" aria-hidden="true"></span>Dashed mark: single-generation baseline. ${spec.rows.some(r => r[1] === null) ? '— In-domain result unavailable.' : ''}</p>
    </section>`;
  }

  function figureArt(kind) {
    if (kind === "transfer") return `<div class="transfer-pair">${transferPlot('crm')}${transferPlot('eops')}</div><p class="visual-footnote">Table 7 · Historical Qwen evaluation. Shared detail scale: 35–55%. Discovery fractions denote task subsets; uncertainty is not included in this website summary.</p>`;
    if (kind === "turnover") return `<div class="outcome-summary"><div><span class="outcome-label">Rescued executions</span><strong>29</strong><p>Baseline failure → selected success</p></div><div class="regression"><span class="outcome-label">Regressions</span><strong>10</strong><p>Baseline success → selected failure</p></div><div class="net-outcome"><span class="outcome-label">Net additional successes</span><strong>+19</strong><p>K = 10 · CRM experiment A</p></div></div><p class="visual-footnote">Whole-episode selection over completed rollouts. 100 recurring CRM tasks; the reported aggregate gain is +6.33 percentage points.</p>`;
    if (kind === "online") return `<div class="comparison-columns"><section><p class="diagram-label">Online guidance</p><h4>Memory informs the next action</h4><p>Retrieve graph evidence during interaction to advise the policy or select a continuation.</p><div class="memory-count"><span>GWM-S discovery rollouts</span><strong>1,573</strong></div><div class="memory-count"><span>GWM-L discovery rollouts</span><strong>14,660</strong></div></section><section><p class="diagram-label">Fixed-candidate replay</p><h4>Isolate the selection decision</h4><ol class="plain-steps"><li>Keep the candidate pool fixed</li><li>Compare selection evidence</li><li>Separate coverage from ranking errors</li></ol><p class="diagram-note">Qwen-generated CRM memory; larger memory helps Gemma in the evaluated GWM-L versus GWM-S comparison.</p></section></div>`;
    if (kind === "guidance") return `<div class="comparison-columns"><section class="failure-example"><p class="diagram-label">Recorded failure pattern</p><h4>Repeated calls, same error</h4><ol class="plain-steps"><li>Attempt the tool call</li><li>Observe the error</li><li>Repeat without fixing the cause</li></ol></section><section><p class="diagram-label">Graph-grounded guidance</p><h4>Verify, then repair</h4><ol class="plain-steps"><li>Check the applicable task rule</li><li>Verify the tool schema and context</li><li>Repair the action before retrying</li></ol></section></div><p class="visual-footnote">Illustrative, anonymized schematic. Historical graph evidence complements task rules and verified live context.</p>`;
    if (kind === "recovery") return `<div class="recovery-map"><div class="recovery-origin"><span class="diagram-label">Current workflow state</span><strong>Evidence identifies a known trap</strong></div><div class="comparison-columns"><section class="failure-example"><p class="diagram-label">Continue the failing branch</p><h4>Retry without a repair</h4><p>Known failing successor → repeated error</p></section><section><p class="diagram-label">Take the repair path</p><h4>Use an observed alternative</h4><p>Verify context → repair action → pursue completion</p></section></div><p class="visual-footnote">The selector retains candidate text and live history. Observed workflow regularities are evidence, not authoritative rules.</p></div>`;
    return `<div class="method-map">
      <section class="method-lane"><div class="lane-heading"><span>01</span><div><h4>Discover workflow memory</h4><p>Offline · successful and failed interactions</p></div></div><div class="method-flow">
        ${flowStep('A','Collect rollouts','Interaction histories and their observed outcomes.')}
        ${flowStep('B','Discover latent states','Embed prefixes, cluster contexts, and link successors.')}
        ${flowStep('C','Build the graph adapter','States, transitions, support, outcomes, and examples.',true)}
      </div></section>
      <div class="method-bridge"><span aria-hidden="true">↓</span> Reuse the same frozen graph at inference time</div>
      <section class="method-lane"><div class="lane-heading"><span>02</span><div><h4>Guide the next action</h4><p>Runtime · policy weights remain unchanged</p></div></div><div class="method-flow">
        ${flowStep('A','Read the live history','Locate the current context in workflow memory.')}
        ${flowStep('B','Retrieve local evidence','Inspect successors, outcomes, and known traps.',true)}
        ${flowStep('C','Advise or select','Supply sparse guidance or rank policy continuations.')}
      </div></section>
      <div class="diagram-bottom"><span>Inspectable evidence</span><span>Frozen policy</span><a href="assets/animation/index.html" target="_blank" rel="noopener">Watch the walkthrough ↗</a></div>
    </div>`;
  }

  function renderTableButtons() {
    const holder = document.getElementById("table-buttons");
    holder.innerHTML = "";
    data.tables.forEach((table, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = table.label;
      button.dataset.table = table.id;
      button.className = index === 0 ? "is-active" : "";
      button.setAttribute("aria-controls", "table-signal data-table");
      button.addEventListener("click", () => { renderTable(table.id); revealSelection(".table-shell"); });
      holder.appendChild(button);
    });
    renderTable(data.tables[0].id);
  }

  function renderTable(id) {
    const tableData = data.tables.find((item) => item.id === id) || data.tables[0];
    document.querySelectorAll("[data-table]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.table === tableData.id);
      button.setAttribute("aria-pressed", String(button.dataset.table === tableData.id));
    });
    const signal = data.tableSignals[tableData.id];
    const signalPanel = document.getElementById("table-signal");
    signalPanel.replaceChildren();
    const label = document.createElement("p");
    label.className = "signal-label";
    label.textContent = "Positive signal";
    const title = document.createElement("h3");
    title.id = "table-signal-title";
    title.textContent = signal.title;
    function evidenceParagraph(text, references, className) {
      const paragraph = document.createElement("p");
      paragraph.className = className;
      paragraph.append(document.createTextNode(text + " "));
      const citation = document.createElement("cite");
      citation.className = "paper-reference";
      references.forEach((ref, index) => {
        if (index) citation.append(document.createTextNode(" · "));
        citation.append(document.createTextNode(`PDF p. ${ref.page}, lines `));
        const range = document.createElement("span");
        range.className = "reference-lines";
        range.textContent = ref.lines;
        citation.appendChild(range);
      });
      paragraph.appendChild(citation);
      return paragraph;
    }
    signalPanel.append(label, title,
      evidenceParagraph(signal.summary, signal.references, "signal-summary"),
      evidenceParagraph(signal.scope, signal.scopeReferences, "signal-scope"));
    const highlighted = new Set(signal.cells.map(([row, column]) => `${row}:${column}`));
    const table = document.getElementById("data-table");
    table.innerHTML = "";
    const caption = document.createElement("caption");
    caption.textContent = `${tableData.label}: ${tableData.title}`;
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    tableData.columns.forEach((column, index) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = column;
      if (index > 0 && tableData.rows.every(row => /^[+\-\d[.]/.test(String(row[index])))) th.className = "numeric";
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    const tbody = document.createElement("tbody");
    tableData.rows.forEach((row, rowIndex) => {
      const tr = document.createElement("tr");
      row.forEach((cell, index) => {
        const el = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) el.scope = "row";
        el.textContent = cell === "-" ? "—" : cell;
        if (index > 0 && /^[+\-\d[.]/.test(String(cell))) el.className = "numeric";
        if (tableData.id === "table2" && index > 0) {
          el.textContent = `${cell} ± ${data.repeatSD[rowIndex][index - 1]}`;
        }
        if (highlighted.has(`${rowIndex}:${index}`)) {
          el.classList.add("positive-signal-cell");
          const value = document.createElement("strong");
          value.textContent = el.textContent;
          el.replaceChildren(value);
        }
        tr.appendChild(el);
      });
      tbody.appendChild(tr);
    });
    table.append(caption, thead, tbody);
    document.getElementById("table-note").textContent = tableData.note;
  }

  function renderTransfer(target = "crm") {
    document.getElementById("transfer-chart").innerHTML = transferPlot(target);
  }

  document.querySelectorAll("[data-transfer-target]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-transfer-target]").forEach((item) => {
        item.classList.toggle("is-active", item === button);
        item.setAttribute("aria-pressed", String(item === button));
      });
      renderTransfer(button.dataset.transferTarget);
      revealSelection(".transfer-lab");
    });
  });

  function renderArchitectureButtons() {
    const holder = document.getElementById("architecture-buttons");
    holder.innerHTML = "";
    data.architecture.forEach((step, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = `<span>0${index + 1}</span><strong>${step.label}</strong>`;
      button.dataset.arch = step.id;
      button.className = index === 0 ? "is-active" : "";
      button.setAttribute("aria-controls", "architecture-detail arch-diagram");
      button.addEventListener("click", () => { renderArchitecture(step.id); revealSelection(".extension-grid"); });
      holder.appendChild(button);
    });
    renderArchitecture(data.architecture[0].id);
  }

  function renderArchitecture(id) {
    const step = data.architecture.find((item) => item.id === id) || data.architecture[0];
    document.querySelectorAll("[data-arch]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.arch === step.id);
      button.setAttribute("aria-pressed", String(button.dataset.arch === step.id));
    });
    document.getElementById("arch-title").textContent = step.title;
    document.getElementById("arch-summary").textContent = step.summary;
    const points = document.getElementById("arch-points");
    points.innerHTML = "";
    step.points.forEach((point) => {
      const li = document.createElement("li");
      li.textContent = point;
      points.appendChild(li);
    });
    document.getElementById("arch-diagram").innerHTML = archArt(step.id);
  }

  function archArt(active) {
    const descriptions = {
      plugin: ['Endpoint plugin','Registers routes and wraps chat serving'],
      registry: ['Graph registry','Validates and loads named adapters'],
      serving: ['Serving proxy','Reads request options and generates candidates'],
      scorer: ['Scorer','Matches live history to workflow states'],
      mediator: ['Mediator','Turns graph evidence into advice or scores'],
      collect: ['Collect and evolve','Records rollouts for future validated graph builds']
    };
    const node = (id) => `<button type="button" class="architecture-node${id === active ? ' selected' : ''}" data-arch-node="${id}" aria-pressed="${id === active}"><strong>${descriptions[id][0]}</strong><span>${descriptions[id][1]}</span></button>`;
    return `<div class="architecture-map">
      <div class="architecture-lane"><p class="diagram-label">Setup & adapters</p><div class="architecture-setup">${node('plugin')}${node('registry')}</div></div>
      <div class="architecture-lane"><div class="architecture-label-row"><p class="diagram-label">GWM-enabled request</p><span>Chat completions + request options</span></div><div class="architecture-request">${node('serving')}${node('scorer')}${node('mediator')}</div></div>
      <div class="architecture-lane architecture-future"><div><p class="diagram-label">Future graph updates</p><p>Finalized experience feeds a later adapter build.</p></div>${node('collect')}</div>
      <div class="diagram-bottom"><span>Model weights unchanged</span><span>No vLLM core patches</span></div>
    </div>`;
  }

  document.getElementById("arch-diagram").addEventListener("click", (event) => {
    const node = event.target.closest("[data-arch-node]");
    if (node) {
      const selected = node.dataset.archNode;
      renderArchitecture(selected);
      document.querySelector(`[data-arch="${selected}"]`).focus({ preventScroll: true });
      revealSelection(".extension-grid");
    }
  });

  function renderNumbers(filter = "") {
    const grid = document.getElementById("number-grid");
    const needle = filter.trim().toLowerCase();
    grid.innerHTML = "";
    data.numbers
      .filter((item) => item.join(" ").toLowerCase().includes(needle))
      .forEach(([value, note, source, tag]) => {
        const article = document.createElement("article");
        article.className = "number-card";
        article.innerHTML = `
          <span>${tag}</span>
          <strong>${value}</strong>
          <p>${note}</p>
          <small>${source}</small>
        `;
        grid.appendChild(article);
      });
    if (!grid.children.length) grid.innerHTML = '<p class="search-empty" role="status">No matching results. Try “success”, “rollout”, or “prefix”.</p>';
  }

  document.getElementById("number-search").addEventListener("input", (event) => {
    renderNumbers(event.target.value);
  });

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.querySelector(button.dataset.copy);
      if (!target) return;
      const text = target.textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const area = document.createElement("textarea");
          area.value = text;
          area.setAttribute("readonly", "");
          area.style.position = "fixed";
          area.style.opacity = "0";
          document.body.appendChild(area);
          area.select();
          document.execCommand("copy");
          area.remove();
        }
      } catch (error) {
        console.warn("Copy failed", error);
        return;
      }
      const previous = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = previous;
      }, 1200);
    });
  });

  // Standalone walkthroughs report their rendered heights to their own parent frame.
  const animationFrames = Array.from(document.querySelectorAll("#gwm-animation, #extension-animation"));
  const figureOne = document.getElementById("gwm-animation");
  let restartPending = false;
  function alignFigureOne() {
    const section = document.getElementById("animation");
    const headerBottom = document.querySelector(".site-header").getBoundingClientRect().bottom;
    window.scrollTo({
      top: Math.max(0, window.scrollY + section.getBoundingClientRect().top - headerBottom - 8),
      behavior: "instant"
    });
    syncWalkthroughVisibility();
  }
  document.querySelector("[data-animation-start]").addEventListener("click", (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    setTab("overview", false);
    if (location.hash !== "#animation") history.pushState(null, "", "#animation");
    document.getElementById("animation").focus({ preventScroll: true });
    restartPending = true;
    alignFigureOne();
    figureOne.contentWindow?.postMessage({ type: "gwm:restart" }, "*");
  });
  function syncWalkthroughVisibility() {
    animationFrames.forEach((frame) => {
      const bounds = frame.getBoundingClientRect();
      const visible = frame.closest(".tab-panel")?.classList.contains("is-active")
        && bounds.bottom > 0 && bounds.top < window.innerHeight;
      frame.contentWindow?.postMessage({ type: "gwm:visibility", visible: Boolean(visible) }, "*");
    });
  }
  window.addEventListener("message", (event) => {
    const frame = animationFrames.find(item => event.source === item.contentWindow);
    if (!frame || !["gwm:resize", "gwm:restarted"].includes(event.data?.type)) return;
    const height = Number(event.data.height);
    if (!Number.isFinite(height) || height < 200 || height > 4000) return;
    frame.style.height = event.data.compact
      ? `${Math.ceil(height) + 2}px`
      : `${Math.max(480, Math.round(window.innerHeight * 0.75))}px`;
    if (frame === figureOne && event.data.type === "gwm:restarted" && restartPending) {
      restartPending = false;
      // Let compact-view resizing and browser scroll anchoring settle before alignment.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (location.hash === "#animation") alignFigureOne();
      }));
    }
  });
  animationFrames.forEach((frame) => {
    frame.addEventListener("load", () => {
      frame.contentWindow.postMessage({ type: "gwm:measure" }, "*");
      syncWalkthroughVisibility();
      if (frame === figureOne && restartPending && location.hash === "#animation") {
        frame.contentWindow.postMessage({ type: "gwm:restart" }, "*");
      } else if (frame === figureOne && location.hash === "#animation") {
        alignFigureOne();
      }
    });
  });
  const frameObserver = new IntersectionObserver(syncWalkthroughVisibility, { threshold: [0, 0.05] });
  animationFrames.forEach(frame => frameObserver.observe(frame));

  renderFigureButtons();
  renderTableButtons();
  renderTransfer("crm");
  renderArchitectureButtons();
  renderNumbers();
  setTab(location.hash.replace("#", "") || "overview", false);
  if (location.hash === "#animation") alignFigureOne();
})();
