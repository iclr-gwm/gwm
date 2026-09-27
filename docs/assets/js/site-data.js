window.GWM_SITE_DATA = {
  figures: [
    {
      id: "fig1",
      label: "Figure 1",
      title: "Graph World Model construction and use",
      summary:
        "Successful and failed offline rollouts form workflow memory. At inference time the same frozen graph can produce advice or select the better continuation while policy weights stay fixed.",
      facts: [
        ["Discovery", "Embed interaction prefixes, cluster states, mine successors."],
        ["Runtime", "Retrieve state evidence, warn on traps, rank candidates."],
        ["Constraint", "The graph and policy weights remain fixed during use."]
      ],
      art: "pipeline"
    },
    {
      id: "fig2",
      label: "Figure 2",
      title: "Cross-benchmark workflow reuse",
      summary:
        "Qwen-derived graphs are exchanged across CRM and EOPS. All eight cross-benchmark configurations exceed their target single-generation baselines.",
      facts: [
        ["CRM baseline", "46.7%"],
        ["EOPS baseline", "36.7%"],
        ["Matched comparisons", "In-domain graphs lead in three of four populated matches."]
      ],
      art: "transfer"
    },
    {
      id: "fig3",
      label: "Figure 3",
      title: "Historical CRM whole-episode selection",
      summary:
        "CRM experiment A at K = 10 yields 29 rescues and 10 regressions relative to baseline, a net gain of 19 successful task executions.",
      facts: [
        ["Tasks", "100 recurring CRM tasks"],
        ["Net gain", "+6.33 percentage points"],
        ["Interface", "Whole-episode selection among completed rollouts."]
      ],
      art: "turnover"
    },
    {
      id: "fig4",
      label: "Figure 4",
      title: "Online guidance and replay evidence",
      summary:
        "The figure contrasts active online guidance with fixed-candidate replay. Larger CRM memory improves Gemma in the evaluated GWM-L versus GWM-S comparison.",
      facts: [
        ["GWM-S memory", "1,573 Qwen-generated CRM discovery rollouts"],
        ["GWM-L memory", "14,660 Qwen-generated CRM discovery rollouts"],
        ["Replay role", "Keeps candidate pools fixed to inspect selection evidence."]
      ],
      art: "online"
    },
    {
      id: "fig5",
      label: "Figure 5",
      title: "Actionable guidance beside a recorded failure",
      summary:
        "A qualitative example shows how graph retry evidence can be combined with task policy and verified schema context. The website keeps the example schematic and anonymized.",
      facts: [
        ["Guidance role", "Warn on repeated failing calls and redirect toward verification."],
        ["Privacy choice", "No names, record identifiers, or private schema content shown here."],
        ["Status", "Illustrative, not a recovered runtime output."]
      ],
      art: "guidance"
    },
    {
      id: "fig6",
      label: "Figure 6",
      title: "Graph-guided recovery alternatives",
      summary:
        "The schematic recovery graph contrasts continuing a failing branch with taking an evidence-backed repair path.",
      facts: [
        ["Graph signal", "Observed successors plus outcome statistics."],
        ["Selector signal", "Candidate text and live history remain available to recover details."],
        ["Limit", "Observed workflow regularities are not authoritative rules."]
      ],
      art: "recovery"
    }
  ],

  tables: [
    {
      id: "table2",
      label: "Table 2",
      title: "Primary success and selection controls",
      note:
        "Mean task success (%) ± sample SD across three repeats of 100 CRM or 80 EOPS tasks; gain is in percentage points. For the primary CRM configurations, uniform selection is the exact expectation over each configuration’s candidate pool. Baselines reuse earlier executions; inference budgets differ. SDs are not confidence intervals.",
      columns: ["Setting", "Baseline", "Uniform", "Majority", "GWM/control", "Gain (pp)"],
      rows: [
        ["Gemma / CRM GWM-L, K = 8", "40.00", "40.50", "45.00", "52.33", "+12.33"],
        ["Gemma / CRM GWM-S, K = 8", "40.00", "39.75", "44.33", "47.33", "+7.33"],
        ["Qwen / CRM GWM-M, K = 10", "46.00", "44.63", "50.33", "52.33", "+6.33"],
        ["Qwen / EOPS joint selection, K = 8", "40.00", "37.08", "36.25", "40.83", "+0.83"],
        ["Qwen / CRM graph-free vote, K = 10", "46.00", "44.07", "48.67", "48.67", "+2.67"]
      ]
    },
    {
      id: "table3",
      label: "Table 3",
      title: "Eventual-success prediction",
      note:
        "Observed prefixes from held-out CRM tasks. Node outcome statistics carry the strongest outcome signal by AUROC.",
      columns: ["Predictor", "AUROC", "95% CI", "Log loss", "Brier", "ECE"],
      rows: [
        ["Frequency", "0.500", "[0.500, 0.500]", "0.481", "0.151", "0.061"],
        ["Position", "0.570", "[0.562, 0.579]", "0.467", "0.146", "0.019"],
        ["Current action", "0.475", "[0.465, 0.484]", "0.471", "0.147", "0.008"],
        ["Node outcome statistics", "0.745", "[0.725, 0.765]", "0.418", "0.136", "0.062"]
      ]
    },
    {
      id: "table4",
      label: "Table 4",
      title: "Discovery corpora and graph sizes",
      note:
        "Historical Qwen discovery-ratio inventory. Ratios are nominal task subsets, not fixed rollout percentages.",
      columns: ["Target", "Ratio", "Rollouts", "Train tasks", "Domains", "States", "Edges"],
      rows: [
        ["CRM", "10%", "1,414", "177", "22", "53", "179"],
        ["CRM", "20%", "2,850", "357", "22", "93", "400"],
        ["CRM", "50%", "7,558", "946", "22", "181", "1,015"],
        ["CRM", "100%", "14,660", "1,836", "22", "346", "2,669"],
        ["EOPS", "10%", "2,170", "53", "8", "112", "1,385"],
        ["EOPS", "20%", "4,013", "98", "8", "212", "2,933"],
        ["EOPS", "50%", "10,279", "251", "8", "527", "9,656"],
        ["EOPS", "100%", "20,864", "510", "8", "1,013", "20,762"]
      ]
    },
    {
      id: "table6",
      label: "Table 6",
      title: "In-domain discovery fractions",
      note:
        "Historical Qwen task success. CRM baseline is 46.7%; EOPS baseline is 36.7% for this discovery experiment.",
      columns: ["Benchmark", "K", "10%", "20%", "50%", "100%"],
      rows: [
        ["CRM", "4", "50.0", "49.7", "50.7", "-"],
        ["CRM", "8", "51.3", "52.0", "48.0", "-"],
        ["CRM", "10", "49.0", "51.3", "49.0", "50.3"],
        ["EOPS", "4", "43.3", "43.8", "42.5", "-"],
        ["EOPS", "8", "41.7", "39.6", "40.0", "-"],
        ["EOPS", "10", "41.3", "43.3", "41.3", "38.8"]
      ]
    },
    {
      id: "table7",
      label: "Table 7",
      title: "Cross-benchmark transfer",
      note:
        "Historical Qwen cross-benchmark task success. Final columns compare against the target baseline and matched in-domain graph.",
      columns: ["Target", "Ratio / K", "In-domain", "Out-of-domain", "vs K = 1", "vs own"],
      rows: [
        ["CRM", "20% / 4", "49.7", "48.3", "+1.6", "-1.4"],
        ["CRM", "20% / 8", "52.0", "49.0", "+2.3", "-3.0"],
        ["CRM", "100% / 4", "-", "48.7", "+2.0", "-"],
        ["CRM", "100% / 8", "-", "49.7", "+3.0", "-"],
        ["EOPS", "20% / 4", "43.8", "41.2", "+4.5", "-2.6"],
        ["EOPS", "20% / 8", "39.6", "42.5", "+5.8", "+2.9"],
        ["EOPS", "100% / 4", "-", "43.8", "+7.1", "-"],
        ["EOPS", "100% / 8", "-", "40.8", "+4.1", "-"]
      ]
    }
  ],

  // Findings and printed PDF margin-line references, checked against the numbered manuscript.
  tableSignals: {
    "table2": {
      "title": "Higher recorded CRM success with GWM",
      "summary": "All three evaluated CRM GWM configurations have higher observed mean success than their recorded direct-generation baselines: gains of 6.33–12.33 percentage points. Gemma GWM-L reaches 52.33%, compared with 40.00% for its baseline.",
      "references": [
        {
          "page": 7,
          "lines": "365–375"
        }
      ],
      "scope": "These are complete evaluated configurations. The table reports three repeats, sample SDs, reused baselines, and differing inference budgets.",
      "scopeReferences": [
        {
          "page": 7,
          "lines": "333–337"
        }
      ],
      "cells": [
        [
          0,
          4
        ],
        [
          0,
          5
        ],
        [
          1,
          4
        ],
        [
          1,
          5
        ],
        [
          2,
          4
        ],
        [
          2,
          5
        ]
      ]
    },
    "table3": {
      "title": "Workflow states retain outcome information",
      "summary": "Node outcome statistics have the highest AUROC (0.745 versus at most 0.570 for the other predictors), the lowest log loss, and the lowest Brier score among the evaluated predictors.",
      "references": [
        {
          "page": 8,
          "lines": "411–417"
        }
      ],
      "scope": "This is eventual-success discrimination. ECE calibration is worse (0.062 versus 0.008 for current action), and later observed prefixes can already contain substantial outcome evidence.",
      "scopeReferences": [
        {
          "page": 8,
          "lines": "418–421"
        }
      ],
      "cells": [
        [
          3,
          1
        ],
        [
          3,
          2
        ],
        [
          3,
          3
        ],
        [
          3,
          4
        ]
      ]
    },
    "table4": {
      "title": "Workflow memory from small discovery subsets",
      "summary": "At the nominal 10% discovery fraction, 1,414 CRM rollouts produce 53 states and 179 successor relations; 2,170 EOPS rollouts produce 112 states and 1,385 successor relations.",
      "references": [
        {
          "page": 19,
          "lines": "1008–1018"
        }
      ],
      "scope": "This table describes graph construction. Ratios refer to task subsets, not rollout percentages; state count alone does not measure the usefulness of retrieved context.",
      "scopeReferences": [
        {
          "page": 19,
          "lines": "996–1002"
        },
        {
          "page": 19,
          "lines": "1024–1025"
        },
        {
          "page": 20,
          "lines": "1026–1029"
        }
      ],
      "cells": [
        [
          0,
          2
        ],
        [
          0,
          5
        ],
        [
          0,
          6
        ],
        [
          4,
          2
        ],
        [
          4,
          5
        ],
        [
          4,
          6
        ]
      ]
    },
    "table6": {
      "title": "Useful guidance at 10% discovery",
      "summary": "The smallest evaluated discovery collections support positive comparisons in both benchmarks. At 10%, CRM success is 49.0–51.3% versus its 46.7% baseline; EOPS is 41.3–43.3% versus 36.7%.",
      "references": [
        {
          "page": 21,
          "lines": "1089–1093"
        }
      ],
      "scope": "The positive means show that the full discovery corpus is unnecessary for useful guidance in these recorded configurations. Each benchmark uses its own experimental baseline.",
      "scopeReferences": [
        {
          "page": 21,
          "lines": "1107–1111"
        }
      ],
      "cells": [
        [
          0,
          2
        ],
        [
          1,
          2
        ],
        [
          2,
          2
        ],
        [
          3,
          2
        ],
        [
          4,
          2
        ],
        [
          5,
          2
        ]
      ]
    },
    "table7": {
      "title": "Workflow memory transfers across benchmarks",
      "summary": "All eight evaluated out-of-domain graph configurations exceed their target’s single-generation baseline. Transferred EOPS reaches 43.8% at 100% / K = 4, matching the highest in-domain rate of 43.8% at 20% / K = 4.",
      "references": [
        {
          "page": 21,
          "lines": "1127–1130"
        },
        {
          "page": 22,
          "lines": "1136–1147"
        }
      ],
      "scope": "Three of the four available equal-ratio comparisons favor the in-domain graph. The paper describes these as comparisons of the evaluated sampling-and-guidance configurations.",
      "scopeReferences": [
        {
          "page": 21,
          "lines": "1127–1130"
        }
      ],
      "cells": [
        [
          0,
          3
        ],
        [
          0,
          4
        ],
        [
          1,
          3
        ],
        [
          1,
          4
        ],
        [
          2,
          3
        ],
        [
          2,
          4
        ],
        [
          3,
          3
        ],
        [
          3,
          4
        ],
        [
          4,
          3
        ],
        [
          4,
          4
        ],
        [
          5,
          3
        ],
        [
          5,
          4
        ],
        [
          6,
          3
        ],
        [
          6,
          4
        ],
        [
          7,
          3
        ],
        [
          7,
          4
        ]
      ]
    }
  },

  // Sample SDs across repeats, in the same order as Table 2 means.
  repeatSD: [
    ["5.00", "1.27", "1.00", "1.53", "5.69"],
    ["5.00", "1.27", "2.08", "3.06", "4.93"],
    ["1.73", "0.51", "0.58", "1.15", "2.52"],
    ["0.00", "5.20", "3.31", "3.15", "3.15"],
    ["1.73", "1.53", "3.79", "3.79", "3.51"]
  ],

  transfer: {
    crm: {
      baseline: 46.7,
      rows: [
        ["20% / K=4", 49.7, 48.3],
        ["20% / K=8", 52.0, 49.0],
        ["100% / K=4", null, 48.7],
        ["100% / K=8", null, 49.7]
      ]
    },
    eops: {
      baseline: 36.7,
      rows: [
        ["20% / K=4", 43.8, 41.2],
        ["20% / K=8", 39.6, 42.5],
        ["100% / K=4", null, 43.8],
        ["100% / K=8", null, 40.8]
      ]
    }
  },

  architecture: [
    {
      id: "plugin",
      label: "Endpoint plugin",
      title: "Registers GWM routes and wraps chat serving",
      summary:
        "The plugin adds `/v1/gwm/*`, optional graph viewer routes, and a batch endpoint. It replaces the stock chat handler with a duck-typed proxy only after vLLM initializes.",
      points: [
        "Discovered through the `vllm.endpoint_plugins` entry point.",
        "No vLLM core files are modified.",
        "Stock requests without `gwm.*` delegate directly to the inner chat handler."
      ]
    },
    {
      id: "registry",
      label: "Graph registry",
      title: "Loads named graph adapters safely",
      summary:
        "Graph bundles are registered by name, validated, cached, and resolved through configured roots. Requests cannot point the server at arbitrary paths.",
      points: [
        "Adapter bundle: manifest, transitions, centroids, metadata, optional examples.",
        "LRU cache keeps scorer and mediator instances warm.",
        "Version activation validates first and then swaps atomically."
      ]
    },
    {
      id: "scorer",
      label: "Scorer",
      title: "Classifies live history into workflow states",
      summary:
        "The scorer embeds a shaped conversation prefix, matches it to graph centroids, and abstains when the distance exceeds the learned threshold.",
      points: [
        "Centroids are normalized 384-dimensional state means.",
        "Trap states use support and failure-lift thresholds.",
        "Abstention is a guardrail for unrelated conversations."
      ]
    },
    {
      id: "mediator",
      label: "Mediator",
      title: "Turns evidence into advice or candidate scores",
      summary:
        "The mediator decides when guidance is warranted, grounds any advice in graph evidence, and ranks candidates with fallback scoring when needed.",
      points: [
        "Advice is sparse by default: trap or retry loop triggers.",
        "Selection supports deduplication, votes, tournament mode, and graph fallback.",
        "Budgets prevent repeated advice in the same episode and state."
      ]
    },
    {
      id: "serving",
      label: "Serving proxy",
      title: "Routes requests by `vllm_xargs`",
      summary:
        "Per-request options choose `advise`, `select`, or `auto`. K greater than 1 selects; K = 1 advises.",
      points: [
        "Select generates `n = k` policy candidates and returns one winning choice.",
        "Advise can regenerate after a rejected candidate, then fail open to the best candidate.",
        "Streaming is buffered because a full candidate must be approved first."
      ]
    },
    {
      id: "collect",
      label: "Collect and evolve",
      title: "Records finalized rollouts for future graph builds",
      summary:
        "Optional collection writes JSONL rollouts and feedback. The evolve path snapshots finalized experience, rebuilds an adapter, and activates validated versions.",
      points: [
        "Explicit feedback can label episodes.",
        "Failed builds leave the active graph untouched.",
        "Current status: rebuild skeleton exists; durable finalize counters remain future work."
      ]
    }
  ],

  numbers: [
    ["+12.33 pp", "Gemma CRM GWM-L improvement over baseline in Table 2.", "Table 2", "success"],
    ["+7.33 pp", "Gemma CRM GWM-S improvement over baseline in Table 2.", "Table 2", "success"],
    ["+6.33 pp", "Qwen CRM GWM-M improvement over baseline in Table 2.", "Table 2", "success"],
    ["100", "CRM task count in the main online comparisons.", "Tables 1 and 2", "task"],
    ["80", "EOPS task count in the main online comparisons.", "Tables 1 and 2", "task"],
    ["65,994", "Observed prefixes in eventual-success prediction.", "Table 3", "prefix"],
    ["1,479", "Held-out CRM tasks excluded from the predictor discovery graph.", "Table 3", "task"],
    ["1,573", "GWM-S Qwen-generated CRM discovery rollouts.", "Tables 2 and 5", "rollout"],
    ["14,660", "GWM-L and several full CRM graph discovery rollout counts.", "Tables 2, 4, and 5", "rollout"],
    ["20,864", "Full EOPS discovery inventory rollout count.", "Table 4", "rollout"],
    ["384", "Centroid vector dimension used by the reference MiniLM embedder.", "Implementation guide", "architecture"],
    ["K = 8", "Candidate count used in Gemma CRM GWM-S/L and EOPS joint selection.", "Table 2", "candidate"],
    ["K = 10", "Candidate count used in Qwen CRM GWM-M.", "Table 2", "candidate"],
    ["1.3x", "Failure-lift threshold for trap-triggered advice.", "Implementation guide", "advice"],
    ["20", "Minimum supporting rollouts for a trap state in the reference implementation.", "Implementation guide", "advice"],
    ["2", "Maximum accepted advice injections per episode by default.", "Implementation guide", "advice"],
    ["2.546M", "Average guidance tokens per 100-task CRM run in evaluated configurations.", "Paper section 6", "cost"],
    ["0", "Required vLLM core modifications.", "README and implementation guide", "architecture"]
  ]
};
