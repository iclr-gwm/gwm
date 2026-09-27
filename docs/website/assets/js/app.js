(function () {
  const data = window.GWM_SITE_DATA;
  const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));

  function setTab(id, updateHash = true) {
    const target = panels.find((panel) => panel.id === id) ? id : "overview";
    tabButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.tab === target);
    });
    panels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.id === target);
    });
    if (updateHash) {
      history.replaceState(null, "", `#${target}`);
    }
    document.getElementById(target).focus({ preventScroll: true });
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });

  document.querySelectorAll("[data-tab-jump]").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tabJump));
  });

  window.addEventListener("hashchange", () => {
    setTab(location.hash.replace("#", ""), false);
  });

  function renderFigureButtons() {
    const holder = document.getElementById("figure-buttons");
    holder.innerHTML = "";
    data.figures.forEach((figure, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${figure.label}: ${figure.title}`;
      button.dataset.figure = figure.id;
      button.className = index === 0 ? "is-active" : "";
      button.addEventListener("click", () => renderFigure(figure.id));
      holder.appendChild(button);
    });
    renderFigure(data.figures[0].id);
  }

  function renderFigure(id) {
    const figure = data.figures.find((item) => item.id === id) || data.figures[0];
    document.querySelectorAll("[data-figure]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.figure === figure.id);
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

  function figureArt(kind) {
    const common =
      '<svg viewBox="0 0 720 360" role="img" aria-label="Sanitized figure schematic">';
    if (kind === "transfer") {
      return `${common}
        <line class="axis" x1="70" y1="296" x2="650" y2="296"></line>
        <line class="axis" x1="70" y1="52" x2="70" y2="296"></line>
        <line class="baseline" x1="92" y1="210" x2="640" y2="210"></line>
        <text x="92" y="196">target baseline</text>
        <g class="bars">
          <rect x="120" y="124" width="54" height="172"></rect>
          <rect x="190" y="148" width="54" height="148"></rect>
          <rect x="292" y="96" width="54" height="200"></rect>
          <rect x="362" y="136" width="54" height="160"></rect>
          <rect x="464" y="118" width="54" height="178"></rect>
          <rect x="534" y="132" width="54" height="164"></rect>
        </g>
        <g class="legend">
          <circle cx="128" cy="46" r="8"></circle><text x="144" y="51">in-domain</text>
          <circle class="alt" cx="266" cy="46" r="8"></circle><text x="282" y="51">cross-benchmark</text>
        </g>
      </svg>`;
    }
    if (kind === "turnover") {
      return `${common}
        <circle class="bubble good" cx="204" cy="168" r="76"></circle>
        <circle class="bubble bad" cx="416" cy="190" r="46"></circle>
        <text class="big" x="160" y="174">29</text>
        <text class="small" x="154" y="210">rescues</text>
        <text class="big" x="390" y="196">10</text>
        <text class="small" x="366" y="232">regressions</text>
        <path class="success-edge" d="M264 168 C320 110 384 124 420 156"></path>
        <text x="286" y="90">K = 10, net +19 successes</text>
      </svg>`;
    }
    if (kind === "online") {
      return `${common}
        <g class="memory-stack">
          <rect x="92" y="90" width="126" height="72"></rect>
          <rect x="112" y="112" width="126" height="72"></rect>
          <rect x="132" y="134" width="126" height="72"></rect>
        </g>
        <text x="100" y="238">small memory</text>
        <g class="memory-stack large">
          <rect x="390" y="64" width="150" height="84"></rect>
          <rect x="414" y="92" width="150" height="84"></rect>
          <rect x="438" y="120" width="150" height="84"></rect>
        </g>
        <text x="424" y="238">large memory</text>
        <path class="success-edge" d="M262 160 C316 112 360 102 410 102"></path>
        <path class="support" d="M258 190 C340 252 438 250 548 190"></path>
      </svg>`;
    }
    if (kind === "guidance") {
      return `${common}
        <rect class="panel-fail" x="72" y="70" width="220" height="210"></rect>
        <rect class="panel-guide" x="424" y="70" width="220" height="210"></rect>
        <text x="108" y="118">repeat call</text>
        <text x="108" y="158">same error</text>
        <text x="108" y="198">same error</text>
        <text x="456" y="118">check rule</text>
        <text x="456" y="158">verify schema</text>
        <text x="456" y="198">repair action</text>
        <path class="trap-edge" d="M292 158 C356 126 388 126 424 158"></path>
        <path class="success-edge" d="M292 210 C360 292 458 292 574 220"></path>
      </svg>`;
    }
    if (kind === "recovery") {
      return `${common}
        <circle class="state" cx="120" cy="180" r="34"></circle>
        <circle class="state trap" cx="302" cy="104" r="46"></circle>
        <circle class="state" cx="302" cy="258" r="42"></circle>
        <circle class="state end" cx="538" cy="180" r="42"></circle>
        <path class="trap-edge" d="M154 170 C208 124 244 106 258 104"></path>
        <path class="success-edge" d="M154 194 C230 256 272 262 302 258 C396 258 466 224 512 194"></path>
        <path class="support" d="M348 104 C430 94 494 118 522 150"></path>
        <text x="92" y="245">state</text>
        <text x="258" y="68">trap</text>
        <text x="256" y="320">repair</text>
        <text x="512" y="244">success</text>
      </svg>`;
    }
    return `${common}
      <circle class="state" cx="92" cy="180" r="38"></circle>
      <circle class="state" cx="244" cy="118" r="42"></circle>
      <circle class="state trap" cx="244" cy="246" r="44"></circle>
      <circle class="state" cx="424" cy="180" r="48"></circle>
      <circle class="state end" cx="604" cy="180" r="40"></circle>
      <path class="support" d="M130 170 C172 128 196 118 202 118"></path>
      <path class="trap-edge" d="M130 194 C174 234 200 246 200 246"></path>
      <path class="success-edge" d="M286 118 C348 116 384 136 404 156"></path>
      <path class="policy-edge" d="M288 246 C358 258 406 238 424 228"></path>
      <path class="success-edge" d="M472 180 C524 180 550 180 564 180"></path>
      <text x="48" y="250">rollouts</text>
      <text x="188" y="70">cluster</text>
      <text x="382" y="112">adapter</text>
      <text x="550" y="248">select</text>
    </svg>`;
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
      button.addEventListener("click", () => renderTable(table.id));
      holder.appendChild(button);
    });
    renderTable(data.tables[0].id);
  }

  function renderTable(id) {
    const tableData = data.tables.find((item) => item.id === id) || data.tables[0];
    document.querySelectorAll("[data-table]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.table === tableData.id);
    });
    const table = document.getElementById("data-table");
    table.innerHTML = "";
    const caption = document.createElement("caption");
    caption.textContent = `${tableData.label}: ${tableData.title}`;
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    tableData.columns.forEach((column) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = column;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    const tbody = document.createElement("tbody");
    tableData.rows.forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((cell, index) => {
        const el = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) el.scope = "row";
        el.textContent = cell;
        tr.appendChild(el);
      });
      tbody.appendChild(tr);
    });
    table.append(caption, thead, tbody);
    document.getElementById("table-note").textContent = tableData.note;
  }

  function renderTransfer(target = "crm") {
    const chart = document.getElementById("transfer-chart");
    const spec = data.transfer[target];
    const max = target === "crm" ? 56 : 47;
    chart.innerHTML = "";
    const baseline = document.createElement("div");
    baseline.className = "baseline-row";
    baseline.innerHTML = `<span>Baseline ${spec.baseline}%</span><i></i>`;
    chart.appendChild(baseline);
    spec.rows.forEach(([label, own, cross]) => {
      const row = document.createElement("div");
      row.className = "transfer-row";
      const ownWidth = own ? Math.max(4, (own / max) * 100) : 0;
      const crossWidth = Math.max(4, (cross / max) * 100);
      row.innerHTML = `
        <strong>${label}</strong>
        <div class="transfer-bars">
          ${own ? `<span class="own" style="width:${ownWidth}%"><b>${own}%</b></span>` : '<span class="empty">in-domain unavailable</span>'}
          <span class="cross" style="width:${crossWidth}%"><b>${cross}%</b></span>
        </div>
      `;
      chart.appendChild(row);
    });
  }

  document.querySelectorAll("[data-transfer-target]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-transfer-target]").forEach((item) => {
        item.classList.toggle("is-active", item === button);
      });
      renderTransfer(button.dataset.transferTarget);
    });
  });

  function renderArchitectureButtons() {
    const holder = document.getElementById("architecture-buttons");
    holder.innerHTML = "";
    data.architecture.forEach((step, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = step.label;
      button.dataset.arch = step.id;
      button.className = index === 0 ? "is-active" : "";
      button.addEventListener("click", () => renderArchitecture(step.id));
      holder.appendChild(button);
    });
    renderArchitecture(data.architecture[0].id);
  }

  function renderArchitecture(id) {
    const step = data.architecture.find((item) => item.id === id) || data.architecture[0];
    document.querySelectorAll("[data-arch]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.arch === step.id);
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
    const nodes = [
      ["plugin", "Plugin", 80, 78],
      ["registry", "Registry", 260, 78],
      ["scorer", "Scorer", 440, 78],
      ["mediator", "Mediator", 260, 228],
      ["serving", "Chat proxy", 80, 228],
      ["collect", "Rollouts", 440, 228]
    ];
    const nodeMarkup = nodes
      .map(([id, label, x, y]) => {
        const cls = id === active ? "arch-node active" : "arch-node";
        return `<g class="${cls}"><rect x="${x}" y="${y}" width="126" height="58" rx="10"></rect><text x="${x + 63}" y="${y + 35}">${label}</text></g>`;
      })
      .join("");
    return `<svg viewBox="0 0 650 350" role="img" aria-label="Architecture diagram">
      <path class="edge support" d="M206 106 H260"></path>
      <path class="edge support" d="M386 106 H440"></path>
      <path class="edge policy-edge" d="M143 228 V136"></path>
      <path class="edge success-edge" d="M206 256 H260"></path>
      <path class="edge success-edge" d="M386 256 H440"></path>
      <path class="edge trap-edge" d="M323 136 V228"></path>
      ${nodeMarkup}
    </svg>`;
  }

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

  const modeToggle = document.getElementById("gwm-mode-toggle");
  modeToggle.addEventListener("change", () => {
    document.getElementById("workflow-graph").classList.toggle("is-baseline", !modeToggle.checked);
    document.getElementById("stage-title").textContent = modeToggle.checked
      ? "Workflow memory is active"
      : "Direct generation is active";
    document.getElementById("stage-copy").textContent = modeToggle.checked
      ? "The graph retrieves local evidence, warns on known traps, and ranks candidate continuations without changing policy weights."
      : "The policy generates directly. Historical workflow evidence is not retrieved for advice or candidate selection.";
  });

  renderFigureButtons();
  renderTableButtons();
  renderTransfer("crm");
  renderArchitectureButtons();
  renderNumbers();
  setTab(location.hash.replace("#", "") || "overview", false);
})();
