# SPDX-License-Identifier: MIT
"""Interactive viewer for mined World-Model transition graphs.

Ported from the benchmark graph viewer. Payload math is
kept close to upstream; the standalone HTTP server is replaced by FastAPI
routes at ``/v1/graph``.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

SKIP = {"START", "END", "noise"}
CANDIDATES = (
    "reports/transitions.json",
    "reports/transform.json",
    "transitions.json",
    "transform.json",
)


def resolve_graph(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for rel in CANDIDATES:
            cand = path / rel
            if cand.is_file():
                return cand
    if path.name == "transform.json":
        alt = path.with_name("transitions.json")
        if alt.is_file():
            return alt
    raise FileNotFoundError(
        f"no transitions.json / transform.json under {path}"
    )


def _top_tools(tool_edges: dict, state: str, k: int = 4) -> list[dict]:
    acts = tool_edges.get(state) or {}
    ranked = sorted(acts.items(), key=lambda kv: -int((kv[1] or {}).get("n") or 0))
    out = []
    for name, info in ranked[:k]:
        info = info or {}
        tools = info.get("tools") or ([name] if name else [])
        label = ", ".join(tools) if tools else "(no-op)"
        out.append(
            {
                "name": label,
                "n": int(info.get("n") or 0),
                "succ": float(info.get("succ") or 0.0),
                "p_action": float(info.get("p_action") or 0.0),
            }
        )
    return out


def _levels(node_ids: set[str], edges: list[dict]) -> dict[str, int]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["source"]].append(e["target"])
    dist: dict[str, int] = {}
    q: deque[str] = deque()
    if "START" in node_ids:
        dist["START"] = 0
        q.append("START")
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    mx = max(dist.values(), default=0)
    for sid in node_ids:
        if sid in dist:
            continue
        dist[sid] = mx + (2 if sid == "END" else 1)
    if "END" in node_ids:
        dist["END"] = max(dist.values()) + 1
    return dist


def _tarjan_sccs(node_ids: set[str], adj: dict[str, list[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    onstack: set[str] = set()
    idx: dict[str, int] = {}
    low: dict[str, int] = {}
    comps: list[list[str]] = []

    def strong(v: str) -> None:
        nonlocal index
        idx[v] = low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in adj.get(v, ()):
            if w not in node_ids:
                continue
            if w not in idx:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in node_ids:
        if v not in idx:
            strong(v)
    return comps


def _bfs_path(adj: dict[str, list[str]], src: str, dst: str) -> list[str]:
    if src == dst:
        return [src]
    prev: dict[str, str] = {}
    q = deque([src])
    seen = {src}
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            prev[v] = u
            if v == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(v)
    return []


def _longest_simple(adj: dict[str, list[str]], src: str, dst: str, limit: int = 14) -> list[str]:
    """Longest simple src→dst path; only for tiny subgraphs (jungles)."""
    best: list[str] = []

    def dfs(u: str, path: list[str], used: set[str]) -> None:
        nonlocal best
        if u == dst and len(path) > len(best):
            best = path.copy()
        if len(path) > limit:
            return
        for v in adj.get(u, ()):
            if v in used:
                continue
            used.add(v)
            path.append(v)
            dfs(v, path, used)
            path.pop()
            used.discard(v)

    dfs(src, [src], {src})
    return best


def _graph_metrics(node_ids: set[str], edges: list[dict], fail_by: dict) -> dict:
    adj: dict[str, list[str]] = defaultdict(list)
    radj: dict[str, list[str]] = defaultdict(list)
    self_loops = 0
    for e in edges:
        a, b = e["source"], e["target"]
        if a == b:
            self_loops += 1
            continue
        adj[a].append(b)
        radj[b].append(a)

    comps = _tarjan_sccs(node_ids, adj)
    cid = {n: i for i, comp in enumerate(comps) for n in comp}
    jungles = [c for c in comps if len(c) >= 2]
    jungle_ids = {n for c in jungles for n in c}

    dag: dict[int, set[int]] = defaultdict(set)
    for a, dests in adj.items():
        for b in dests:
            if a in cid and b in cid and cid[a] != cid[b]:
                dag[cid[a]].add(cid[b])

    start_c = cid.get("START")
    end_c = cid.get("END")
    longest_comp: list[int] = []
    if start_c is not None and end_c is not None:
        memo: dict[int, list[int]] = {}

        def chain(c: int) -> list[int]:
            if c in memo:
                return memo[c]
            best = [c] if c == end_c else []
            for nxt in dag.get(c, ()):
                rest = chain(nxt)
                if rest and 1 + len(rest) > len(best):
                    best = [c] + rest
            memo[c] = best
            return best

        longest_comp = chain(start_c)

    longest_path: list[str] = []
    if longest_comp:
        cur = "START" if "START" in node_ids else comps[longest_comp[0]][0]
        longest_path = [cur]
        for i, c in enumerate(longest_comp):
            members = comps[c]
            if i + 1 < len(longest_comp):
                nxt_members = set(comps[longest_comp[i + 1]])
                exit_n = None
                entry_n = None
                for u in members:
                    for v in adj.get(u, ()):
                        if v in nxt_members:
                            exit_n, entry_n = u, v
                            break
                    if exit_n:
                        break
                if exit_n is None:
                    continue
                local = {x: [y for y in adj.get(x, ()) if y in members] for x in members}
                if cur in members and len(members) <= 12:
                    internal = _longest_simple(local, cur, exit_n)
                else:
                    internal = _bfs_path(local, cur, exit_n)
                if len(internal) < 2:
                    internal = [cur, exit_n] if cur != exit_n else [cur]
                if len(internal) >= 2:
                    longest_path.extend(internal[1:])
                elif longest_path[-1] != exit_n:
                    longest_path.append(exit_n)
                if longest_path[-1] != entry_n:
                    longest_path.append(entry_n)
                cur = entry_n
            else:
                target = "END" if "END" in members else members[0]
                local = {x: [y for y in adj.get(x, ()) if y in members] for x in members}
                if cur in members and len(members) <= 12:
                    internal = _longest_simple(local, cur, target)
                else:
                    internal = _bfs_path(local if cur in members else adj, cur, target)
                if len(internal) >= 2:
                    longest_path.extend(internal[1:])
                elif longest_path[-1] != target:
                    longest_path.append(target)

    shortest_path = _bfs_path(adj, "START", "END") if "START" in node_ids else []
    from_start = set()
    if "START" in node_ids:
        q = deque(["START"])
        from_start.add("START")
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in from_start:
                    from_start.add(v)
                    q.append(v)
    to_end = set()
    if "END" in node_ids:
        q = deque(["END"])
        to_end.add("END")
        while q:
            u = q.popleft()
            for v in radj.get(u, ()):
                if v not in to_end:
                    to_end.add(v)
                    q.append(v)

    n_states = sum(1 for n in node_ids if n not in SKIP)
    out_deg = [len({t for t in adj.get(s, ()) if t != s}) for s in node_ids if s not in SKIP]
    traps = sum(
        1
        for s in node_ids
        if s not in SKIP and float((fail_by.get(s) or {}).get("fail_rate") or 0) >= 0.6
    )
    return {
        "shortest_hops": max(0, len(shortest_path) - 1) if shortest_path else None,
        "shortest_path": shortest_path,
        "longest_hops": max(0, len(longest_path) - 1) if longest_path else None,
        "longest_path": longest_path,
        "jungles": len(jungles),
        "jungle_states": len(jungle_ids),
        "jungle_ids": sorted(jungle_ids),
        "self_loops": self_loops,
        "traps": traps,
        "avg_out": round(sum(out_deg) / max(len(out_deg), 1), 2),
        "unreachable": sum(1 for s in node_ids if s not in SKIP and s not in from_start),
        "no_exit": sum(1 for s in node_ids if s not in SKIP and s not in to_end),
        "n_states": n_states,
    }


def domain_payload(name: str, raw: dict) -> dict:
    p_next = raw.get("P_next") or {}
    fail_by = {r["state"]: r for r in (raw.get("state_fail") or []) if "state" in r}
    tv_by = {r["state"]: r for r in (raw.get("divergence") or []) if "state" in r}
    tool_edges = raw.get("tool_edges") or {}
    node_ids: set[str] = set()
    for src, nxt in p_next.items():
        node_ids.add(src)
        node_ids.update(nxt or {})
    node_ids.update(fail_by)
    node_ids.update(tv_by)

    edges = []
    for t in raw.get("trans_fail") or []:
        a, b = t.get("from"), t.get("to")
        if not a or not b:
            continue
        node_ids.add(a)
        node_ids.add(b)
        edges.append(
            {
                "source": a,
                "target": b,
                "n": int(t.get("n") or 0),
                "p": float((p_next.get(a) or {}).get(b) or 0.0),
                "fail_rate": float(t.get("fail_rate") or 0.0),
                "lift": float(t.get("lift") or 0.0),
                "self_loop": bool(t.get("self_loop") or a == b),
            }
        )
    if not edges:
        for a, nxt in p_next.items():
            for b, p in (nxt or {}).items():
                edges.append(
                    {
                        "source": a,
                        "target": b,
                        "n": 0,
                        "p": float(p or 0.0),
                        "fail_rate": 0.0,
                        "lift": 0.0,
                        "self_loop": a == b,
                    }
                )

    levels = _levels(node_ids, edges)
    nodes = []
    for sid in sorted(node_ids, key=lambda s: (levels.get(s, 99), s)):
        kind = "sentinel" if sid in SKIP else "state"
        info = fail_by.get(sid) or {}
        div = tv_by.get(sid) or {}
        nodes.append(
            {
                "id": sid,
                "label": sid,
                "kind": kind,
                "level": int(levels.get(sid, 1)),
                "n_rollouts": int(info.get("n_rollouts") or 0),
                "fail_rate": None if kind == "sentinel" else float(info.get("fail_rate") or 0.0),
                "lift": None if kind == "sentinel" else float(info.get("lift") or 0.0),
                "tv": float(div.get("tv") or 0.0),
                "visits": int(div.get("visits") or 0),
                "top_tools": _top_tools(tool_edges, sid),
            }
        )
    n_states = sum(1 for n in nodes if n["kind"] == "state")
    metrics = _graph_metrics(node_ids, edges, fail_by)
    jungle = set(metrics["jungle_ids"])
    for n in nodes:
        n["jungle"] = n["id"] in jungle
    return {
        "name": name,
        "n_rollouts": int(raw.get("n_rollouts") or 0),
        "base_fail_rate": float(raw.get("base_fail_rate") or 0.0),
        "n_states": n_states,
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "stats": {k: v for k, v in metrics.items() if k != "jungle_ids"},
    }


def build_payload(transitions_path: Path) -> dict:
    raw = json.loads(transitions_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{transitions_path} is not a domain-keyed transitions object")
    domains = []
    for name in sorted(raw):
        block = raw[name]
        if isinstance(block, dict) and "P_next" in block:
            domains.append(domain_payload(name, block))
    if not domains:
        raise ValueError(f"no domain graphs with P_next in {transitions_path}")
    return {
        "id": "",
        "label": "",
        "source": str(transitions_path),
        "title": transitions_path.parent.parent.name
        if transitions_path.parent.name == "reports"
        else transitions_path.stem,
        "domains": domains,
    }



API_BASE = "/v1/graph"


def _template_text() -> str:
    path = Path(__file__).with_name("show_graph.html")
    try:
        from importlib.resources import files

        return files("vllm_gwm.viz").joinpath("show_graph.html").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return path.read_text(encoding="utf-8")


def render_html(bundle: dict) -> str:
    template = _template_text()
    blob = json.dumps(bundle, separators=(",", ":"))
    blob = blob.replace("<", "\\u003c")
    marker = "<!--GRAPH_JSON-->"
    if marker not in template:
        raise RuntimeError("show_graph.html missing <!--GRAPH_JSON--> slot")
    return template.replace(marker, blob, 1)


def public_catalog(entries: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "_path"} for e in entries]


def _default_id(entries: list[dict], preferred: str | None) -> str:
    if preferred:
        return preferred
    for e in entries:
        if e["available"]:
            return e["id"]
    return ""


def entries_from_registered(registered: Mapping[str, Path]) -> list[dict[str, Any]]:
    """Build catalog entries from GWM adapter name→path mappings."""
    entries: list[dict[str, Any]] = []
    for name, path in registered.items():
        resolved = None
        try:
            resolved = resolve_graph(Path(path))
        except FileNotFoundError:
            resolved = None
        entries.append(
            {
                "id": str(name),
                "label": str(name),
                "bench": "",
                "tag": "",
                "available": resolved is not None,
                "source": str(resolved) if resolved else None,
                "_path": resolved,
            }
        )
    return entries


class GraphViewer:
    """Lazy catalog + payload cache for the ``/v1/graph`` UI."""

    def __init__(
        self,
        entries: list[dict],
        default_id: str = "",
        *,
        api_base: str = API_BASE,
    ):
        self.entries = entries
        self.by_id = {e["id"]: e for e in entries}
        self.default_id = default_id or _default_id(entries, None)
        self.api_base = api_base.rstrip("/")
        self.cache: dict[str, dict] = {}

    @classmethod
    def from_registered(
        cls, registered: Mapping[str, Path], *, api_base: str = API_BASE
    ) -> "GraphViewer":
        entries = entries_from_registered(registered)
        return cls(entries, api_base=api_base)

    def payload_for(self, gid: str) -> dict:
        entry = self.by_id.get(gid)
        if entry is None or not entry.get("available") or entry.get("_path") is None:
            raise KeyError(gid)
        if gid not in self.cache:
            data = build_payload(entry["_path"])
            data["id"] = gid
            data["label"] = entry["label"]
            data["title"] = entry["label"]
            self.cache[gid] = data
        return self.cache[gid]

    def catalog(self) -> dict[str, Any]:
        return {
            "graphs": public_catalog(self.entries),
            "default": self.default_id,
            "api_base": self.api_base,
        }

    def index_html(self) -> str:
        inline: dict[str, dict] = {}
        if self.default_id:
            try:
                inline[self.default_id] = self.payload_for(self.default_id)
            except (KeyError, OSError, ValueError):
                pass
        bundle = {
            "catalog": public_catalog(self.entries),
            "default": self.default_id,
            "graphs": inline,
            "api_base": self.api_base,
        }
        return render_html(bundle)
