# SPDX-License-Identifier: MIT
"""vLLM GWM CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from vllm_gwm.apis.factory import list_api_presets
from vllm_gwm.build import pipeline
from vllm_gwm.build.layout import graphs_dir
from vllm_gwm.build.mining.embed import DEFAULT_BATCH_SIZE, DEFAULT_EMBED_MODEL
from vllm_gwm.runtime.adapter import GraphAdapter

_SHOW_GRAPH_FLAGS = ("--gwm-show-graph", "--gwm-graph-viz")

_API_FLAGS = {
    "--backend",
    "--api-class",
    "--api-preset",
    "--api-base",
    "--judge-api-class",
    "--judge-api-preset",
    "--judge-api-base",
    "--judge-model",
    "--judge-api-provider",
    "--api-provider",
    "--gemini-region",
    "--list-api-presets",
    "--host",
    "--port",
}


def consume_gwm_serve_args(
    vllm_args: list[str],
    modules: list[str],
    show_graph: bool,
) -> tuple[list[str], list[str], bool]:
    """Pull GWM flags out of ``vllm serve`` remainder so vLLM does not reject them."""
    out: list[str] = []
    i = 0
    while i < len(vllm_args):
        a = vllm_args[i]
        if a in _SHOW_GRAPH_FLAGS:
            show_graph = True
            i += 1
            continue
        if a.startswith("--gwm-show-graph=") or a.startswith("--gwm-graph-viz="):
            show_graph = a.split("=", 1)[1].strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            i += 1
            continue
        if a == "--gwm-modules":
            i += 1
            if i < len(vllm_args):
                modules.append(vllm_args[i])
                i += 1
            continue
        if a.startswith("--gwm-modules="):
            modules.append(a.split("=", 1)[1])
            i += 1
            continue
        out.append(a)
        i += 1
    return out, modules, show_graph


def consume_api_serve_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse API-mode flags from serve remainder; return namespace + leftover vLLM args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default=os.getenv("VLLM_GWM_BACKEND", "vllm"))
    parser.add_argument("--api-class", default=os.getenv("VLLM_GWM_API_CLASS", "openai"))
    parser.add_argument("--api-preset", default=os.getenv("VLLM_GWM_API_PRESET", ""))
    parser.add_argument("--api-base", default=os.getenv("VLLM_GWM_API_BASE", ""))
    parser.add_argument("--judge-api-class", default=os.getenv("VLLM_GWM_JUDGE_API_CLASS", ""))
    parser.add_argument(
        "--judge-api-preset", default=os.getenv("VLLM_GWM_JUDGE_API_PRESET", "")
    )
    parser.add_argument(
        "--judge-api-base", default=os.getenv("VLLM_GWM_JUDGE_BASE_URL", "")
    )
    parser.add_argument("--judge-model", default=os.getenv("VLLM_GWM_JUDGE_MODEL", ""))
    parser.add_argument(
        "--judge-api-provider",
        default=os.getenv("VLLM_GWM_JUDGE_API_PROVIDER", ""),
    )
    parser.add_argument(
        "--api-provider", default=os.getenv("VLLM_GWM_API_PROVIDER", "")
    )
    parser.add_argument(
        "--gemini-region", default=os.getenv("VLLM_GWM_GEMINI_REGION", "")
    )
    parser.add_argument("--list-api-presets", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gwm-show-graph", action="store_true")
    parser.add_argument("--gwm-graph-viz", action="store_true")
    parser.add_argument("--gwm-modules", action="append", default=[])
    parser.add_argument("--model", default=os.getenv("VLLM_GWM_POLICY_MODEL", ""))
    known, rest = parser.parse_known_args(argv)
    # --host/--port/--model are meaningful to BOTH backends: the API server needs
    # them, and so does vLLM's api_server. parse_known_args consumes them, and
    # the vLLM path never re-adds them — so `serve --port 8790` silently bound
    # vLLM's default :8000 instead. Hand them back so the vLLM path can forward
    # them; API mode reads them off the namespace and ignores the leftovers.
    rest = _readd_shared_flags(argv, rest)
    return known, rest


_SHARED_FLAGS = ("--host", "--port", "--model")


def _readd_shared_flags(argv: list[str], rest: list[str]) -> list[str]:
    """Re-append --host/--port/--model to the vLLM passthrough if given."""
    out = list(rest)
    for i, tok in enumerate(argv):
        name, eq, inline = tok.partition("=")
        if name not in _SHARED_FLAGS or name in out:
            continue
        if eq:
            out += [name, inline]
        elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            out += [name, argv[i + 1]]
    return out


def normalize_api_server_args(vllm_args: list[str]) -> list[str]:
    """Prepare argv for ``vllm.entrypoints.openai.api_server``."""
    args = list(vllm_args or [])
    while args and args[0] == "--":
        args = args[1:]
    if not args:
        return args
    if args[0] == "--model":
        return args
    if not args[0].startswith("-"):
        return ["--model", args[0], *args[1:]]
    return args


def _print_api_presets() -> None:
    rows = list_api_presets()
    if not rows:
        print("No API presets (internal_api module not installed).")
        return
    print("API presets:")
    for row in rows:
        print(
            f"  {row['name']:22} provider={row['provider']:6} "
            f"model={row['model']}"
        )
        print(f"    {row['endpoint']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vllm-gwm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Launch vLLM or API backend with GWM")
    serve.add_argument("--gwm-modules", action="append", default=[])
    serve.add_argument(
        "--gwm-show-graph",
        "--gwm-graph-viz",
        dest="gwm_show_graph",
        action="store_true",
        help="Serve the transition-graph viewer at GET /v1/graph",
    )
    serve.add_argument("vllm_args", nargs=argparse.REMAINDER)

    build = sub.add_parser("build", help="Build a graph adapter from rollouts")
    build.add_argument("--input", required=True,
                       help="rollouts jsonl(.gz): collection-store snapshot or "
                            "conversation_flow rollouts")
    build.add_argument("--adapter", required=True)
    build.add_argument("--output", default="")
    build.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    build.add_argument("--embed-device", default=None,
                       help="cpu | cuda | cuda:N (default: sentence-transformers auto)")
    build.add_argument("--embed-batch", type=int, default=DEFAULT_BATCH_SIZE)
    build.add_argument("--trust-remote-code", action="store_true",
                       help="required by embedders shipping custom modelling code")
    build.add_argument("--reuse-steps", action="store_true",
                       help="reuse existing out/steps*.jsonl.gz (e.g. re-embed only)")
    build.add_argument("--watched-tools", default="",
                       help="comma/space separated tools for the precheck gates")
    build.add_argument("--discover-args", default=None,
                       help="e.g. '--within-domain --min-cluster-size 50'")

    inspect = sub.add_parser("inspect", help="Validate a graph adapter bundle")
    inspect.add_argument("path")

    imp = sub.add_parser("import", help="Import a graph bundle into data root")
    imp.add_argument("source")
    imp.add_argument("--name", required=True)

    sub.add_parser("feedback", help="Record explicit rollout feedback")
    rollback = sub.add_parser("rollback", help="Activate a prior graph version")
    rollback.add_argument("--adapter", required=True)
    rollback.add_argument("--version-path", required=True)

    # ``serve`` forwards everything after the subcommand verbatim. argparse's
    # REMAINDER cannot do this when the first token is an option (it tries to
    # match ``--backend`` against the serve parser and errors), so split argv
    # by hand and let consume_*_serve_args own the flags.
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        args = argparse.Namespace(
            cmd="serve", gwm_modules=[], gwm_show_graph=False, vllm_args=argv[1:]
        )
    else:
        args = parser.parse_args()
    if args.cmd == "serve":
        raw = list(args.vllm_args or [])
        while raw and raw[0] == "--":
            raw = raw[1:]
        api_args, vllm_args = consume_api_serve_args(raw)
        modules = list(args.gwm_modules or []) + list(api_args.gwm_modules or [])
        show_graph = bool(args.gwm_show_graph or api_args.gwm_show_graph or api_args.gwm_graph_viz)

        if api_args.list_api_presets:
            _print_api_presets()
            return

        if api_args.backend.strip().lower() == "api":
            if vllm_args and any(
                a.startswith("--") and a.split("=", 1)[0] not in _API_FLAGS
                for a in vllm_args
                if a not in ("--",)
            ):
                print(
                    "warning: vLLM engine flags are ignored in --backend api mode",
                    file=sys.stderr,
                )
            api_args.gwm_modules = modules
            api_args.gwm_show_graph = show_graph
            if not api_args.model and not api_args.api_preset:
                print(
                    "error: --backend api requires --model or --api-preset",
                    file=sys.stderr,
                )
                sys.exit(2)
            if modules:
                os.environ["VLLM_GWM_MODULES"] = ",".join(modules)
            if show_graph:
                os.environ["VLLM_GWM_SHOW_GRAPH"] = "1"
            from vllm_gwm.server import run_server

            run_server(api_args)
            return

        env = os.environ
        vllm_args, modules2, show_graph2 = consume_gwm_serve_args(
            vllm_args, modules, show_graph
        )
        modules = modules2
        show_graph = show_graph2
        vllm_args = normalize_api_server_args(vllm_args)
        if modules:
            env["VLLM_GWM_MODULES"] = ",".join(modules)
        if show_graph:
            env["VLLM_GWM_SHOW_GRAPH"] = "1"
        env.setdefault("VLLM_PLUGINS", "gwm")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(vllm_args)
        subprocess.run(cmd, check=False)
    elif args.cmd == "build":
        out = Path(args.output or graphs_dir() / args.adapter / "build")
        # Stage 0 (harvest) is benchmark-specific: ingest the given rollouts into
        # the workdir, then run the in-plugin mining DAG (no shelling out).
        pipeline.ingest_rollouts(args.input, out)
        pipeline.build_graph(
            out,
            embed_model=args.embed_model,
            embed_device=args.embed_device,
            embed_batch=args.embed_batch,
            trust_remote_code=args.trust_remote_code,
            reuse_steps=args.reuse_steps,
            watched_tools=args.watched_tools,
            discover_args=args.discover_args,
        )
        print(f"built {out}")
    elif args.cmd == "inspect":
        info = GraphAdapter.validate(Path(args.path))
        print(info)
    elif args.cmd == "import":
        dst = graphs_dir() / args.name
        dst.mkdir(parents=True, exist_ok=True)
        src = Path(args.source)
        if src.is_dir():
            import shutil

            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        print(f"imported {src} -> {dst}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
