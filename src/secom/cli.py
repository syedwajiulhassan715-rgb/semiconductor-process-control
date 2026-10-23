"""Command-line interface: ``secom fetch | run | stages | monitor | validate``."""

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
    from .fetch import fetch, fetch_tep

    raw = _root(args) / "data" / "raw"
    if args.dataset == "tep":
        fetch_tep(raw)
    else:
        fetch(raw)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .tep_validation import (
        TEPValidationConfig,
        run_tep_validation,
        write_figures,
        write_results,
    )

    root = _root(args)
    cfg = TEPValidationConfig()
    overrides = {
        k: v
        for k, v in (("alpha", args.alpha), ("n_components", args.n_components))
        if v is not None
    }
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    start = time.perf_counter()
    try:
        results, fig_data = run_tep_validation(root, cfg, with_secom=not args.no_secom)
    except (FileNotFoundError, ValueError) as exc:
        print(f"secom: {exc}", file=sys.stderr)
        return 2
    out = write_results(results, root / "reports" / "validation" / "tep_results.json")
    print(f"wrote {out}")
    if not args.no_figures:
        figures_dir = args.figures_dir or root / "reports" / "figures"
        for path in write_figures(results, fig_data, figures_dir, fmt=args.figure_format):
            print(f"wrote {path}")
    cmp_ = results["benchmark_comparison"]
    print(
        f"done in {time.perf_counter() - start:.1f}s: mean |detection - benchmark| "
        f"T2 {cmp_['t2']['mean_abs_diff']:.3f}, SPE {cmp_['spe']['mean_abs_diff']:.3f}"
    )
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
    if args.export is not None:
        from .monitor_export import export_static

        written = export_static(args.export, session.design, session.runs, speed=args.speed)
        for path in written:
            print(f"wrote {path}")
        print("static monitor: serve that folder with any web server and open index.html")
        return 0
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

    p_fetch = sub.add_parser("fetch", parents=[common], help="download raw data into data/raw/")
    p_fetch.add_argument(
        "--dataset",
        choices=("secom", "tep"),
        default="secom",
        help="secom (default) or tep, the Tennessee Eastman validation data (data/raw/tep/)",
    )
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
    p_monitor.add_argument(
        "--export",
        metavar="DIR",
        type=Path,
        default=None,
        help="write a static, server-free copy of the monitor into DIR instead of serving",
    )
    p_monitor.add_argument("-v", "--verbose", action="store_true", help="log every request")
    p_monitor.set_defaults(func=_cmd_monitor)

    p_validate = sub.add_parser(
        "validate",
        parents=[common],
        help="validate the monitoring code on a benchmark with known faults",
    )
    p_validate.add_argument("benchmark", choices=("tep",), help="tep: Tennessee Eastman")
    p_validate.add_argument("--alpha", type=float, default=None, help="per-chart alpha")
    p_validate.add_argument(
        "--n-components", type=int, default=None, help="PCA components (default 11)"
    )
    p_validate.add_argument(
        "--no-secom", action="store_true", help="skip the SECOM calibration comparison"
    )
    p_validate.add_argument("--no-figures", action="store_true", help="skip writing figures")
    p_validate.add_argument(
        "--figure-format", choices=FIGURE_FORMATS, default="png", help="png (default) or svg"
    )
    p_validate.add_argument(
        "--figures-dir", type=Path, default=None, help="default: reports/figures/"
    )
    p_validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
