"""Command-line interface: ``secom fetch | run | stages | monitor``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import FIGURE_FORMATS, ConfigError, load_config

DEFAULT_CONFIG = Path("config") / "default.yaml"


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root).resolve() if args.root else Path.cwd()


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import fetch

    fetch(_root(args) / "data" / "raw")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Paths, PipelineContext, PipelineError, run_pipeline

    root = _root(args)
    config_path = args.config
    if config_path is None and (root / DEFAULT_CONFIG).exists():
        config_path = root / DEFAULT_CONFIG
    try:
        cfg = load_config(config_path)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"secom: {exc}", file=sys.stderr)
        return 2
    if args.figure_format:
        cfg = dataclasses.replace(
            cfg, figures=dataclasses.replace(cfg.figures, format=args.figure_format)
        )

    ctx = PipelineContext(config=cfg, paths=Paths.from_root(root, figures=args.figures_dir))
    start = time.perf_counter()
    try:
        ran = run_pipeline(ctx, only=args.only, skip_figures=args.no_figures)
    except (PipelineError, FileNotFoundError) as exc:
        print(f"secom: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - start
    print(f"done: {len(ran)} stage(s) in {elapsed:.1f}s")
    if "export" not in ran:
        print("note: export did not run, so results.json and the CSVs were not written")
    return 0


def _cmd_stages(args: argparse.Namespace) -> int:
    from .pipeline import STAGES

    width = max(len(s.NAME) for s in STAGES)
    for i, stage in enumerate(STAGES, start=1):
        doc = (stage.__doc__ or "").strip().splitlines()
        print(f"{i:>2}. {stage.NAME:<{width}}  {doc[0] if doc else ''}")
        if args.verbose:
            print(f"    {'':<{width}}  requires: {', '.join(stage.REQUIRES) or '-'}")
            produces = getattr(stage, "PRODUCES", ())
            print(f"    {'':<{width}}  produces: {', '.join(produces) or '-'}")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from .monitor import session_from_outputs
    from .monitor_server import serve

    if args.speed <= 0:
        print("secom: --speed must be positive", file=sys.stderr)
        return 2
    try:
        session = session_from_outputs(_root(args))
    except FileNotFoundError as exc:
        print(f"secom: {exc}", file=sys.stderr)
        return 2
    try:
        server = serve(
            session, host=args.host, port=args.port, speed=args.speed, quiet=not args.verbose
        )
    except OSError as exc:
        print(f"secom: cannot listen on {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 2
    print(f"replay monitor: {len(session.runs)} runs, watching {session.design.watch_sensor}")
    print(f"open {server.url}  (Ctrl+C to stop)")
    if args.open:
        import webbrowser

        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .pipeline import stage_names

    parser = argparse.ArgumentParser(
        prog="secom",
        description="Statistical process control analysis of the UCI SECOM fab dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root holding data/ and reports/ (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_fetch = sub.add_parser("fetch", parents=[common], help="download SECOM into data/raw/")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_run = sub.add_parser("run", parents=[common], help="run the analysis pipeline")
    p_run.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config (default: <root>/{DEFAULT_CONFIG.as_posix()} if present)",
    )
    p_run.add_argument(
        "--only",
        nargs="+",
        choices=stage_names(),
        metavar="STAGE",
        help="run these stages and the upstream stages they depend on",
    )
    p_run.add_argument("--no-figures", action="store_true", help="skip writing figures")
    p_run.add_argument(
        "--figure-format",
        choices=FIGURE_FORMATS,
        default=None,
        help="override figures.format (svg needs no compiled matplotlib backend)",
    )
    p_run.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="write figures here instead of reports/figures/",
    )
    p_run.set_defaults(func=_cmd_run)

    p_stages = sub.add_parser("stages", help="list the pipeline stages in order")
    p_stages.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show the artifacts each stage reads and writes",
    )
    p_stages.set_defaults(func=_cmd_stages)

    p_monitor = sub.add_parser(
        "monitor",
        parents=[common],
        help="replay the runs through the recommended monitors in a local web page",
    )
    p_monitor.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: %(default)s)"
    )
    p_monitor.add_argument("--port", type=int, default=8050, help="port (default: %(default)s)")
    p_monitor.add_argument(
        "--speed",
        type=float,
        default=20.0,
        metavar="RUNS_PER_SEC",
        help="initial playback speed (default: %(default)s)",
    )
    p_monitor.add_argument("--open", action="store_true", help="open the page in a browser")
    p_monitor.add_argument("-v", "--verbose", action="store_true", help="log every request")
    p_monitor.set_defaults(func=_cmd_monitor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
