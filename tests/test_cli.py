"""CLI subcommands with the heavy work (network, TEP, web server) replaced by fakes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import secom.fetch
import secom.monitor
import secom.monitor_server
import secom.tep_validation
from _synthetic import patch_shape, write_config, write_raw
from secom import __version__
from secom.cli import build_parser, main

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["monitor"])
    assert (args.host, args.port, args.speed) == ("127.0.0.1", 8050, 20.0)


# fetch


@pytest.mark.parametrize(("dataset", "called"), [("secom", "fetch"), ("tep", "fetch_tep")])
def test_fetch_dispatches_by_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: str, called: str
) -> None:
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(secom.fetch, "fetch", lambda raw: calls.append(("fetch", raw)))
    monkeypatch.setattr(secom.fetch, "fetch_tep", lambda raw: calls.append(("fetch_tep", raw)))
    assert main(["fetch", "--root", str(tmp_path), "--dataset", dataset]) == 0
    assert calls == [(called, tmp_path.resolve() / "data" / "raw")]


def test_fetch_defaults_to_the_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(secom.fetch, "fetch", calls.append)
    monkeypatch.chdir(tmp_path)
    assert main(["fetch"]) == 0
    assert calls == [Path.cwd() / "data" / "raw"]


# run


def test_run_partial_pipeline_on_synthetic_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_shape(monkeypatch)
    write_raw(tmp_path / "data" / "raw")
    # picked up as <root>/config/default.yaml without --config
    write_config(tmp_path / "config" / "default.yaml")
    code = main(
        [
            "run",
            "--root",
            str(tmp_path),
            "--only",
            "changepoint",
            "--no-figures",
            "--figure-format",
            "svg",
            "--figures-dir",
            str(tmp_path / "figs"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "done: 4 stage(s)" in out
    assert "note: export did not run" in out
    assert not (tmp_path / "reports" / "results.json").exists()
    assert not list((tmp_path / "figs").glob("*"))


def test_run_without_raw_data_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = write_config(tmp_path / "fast.yaml")
    code = main(["run", "--root", str(tmp_path), "--config", str(cfg), "--only", "load"])
    assert code == 2
    assert "secom.data" in capsys.readouterr().err


def test_run_with_missing_config_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--root", str(tmp_path), "--config", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "config file not found" in capsys.readouterr().err


# validate


def _fake_validation(
    monkeypatch: pytest.MonkeyPatch, *, error: Exception | None = None
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def run(root: Path, cfg: Any, *, with_secom: bool) -> tuple[dict, dict]:
        if error is not None:
            raise error
        seen.update(cfg=cfg, with_secom=with_secom)
        cmp_ = {"t2": {"mean_abs_diff": 0.01}, "spe": {"mean_abs_diff": 0.02}}
        return {"benchmark_comparison": cmp_}, {}

    def write_results(results: dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results), encoding="utf-8")
        return path

    def write_figures(results: dict, fig_data: dict, out: Path, *, fmt: str) -> list[Path]:
        seen.update(figures_dir=out, fmt=fmt)
        return [Path(out) / f"fig_tep.{fmt}"]

    monkeypatch.setattr(secom.tep_validation, "run_tep_validation", run)
    monkeypatch.setattr(secom.tep_validation, "write_results", write_results)
    monkeypatch.setattr(secom.tep_validation, "write_figures", write_figures)
    return seen


def test_validate_applies_overrides_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _fake_validation(monkeypatch)
    code = main(
        [
            "validate",
            "tep",
            "--root",
            str(tmp_path),
            "--alpha",
            "0.02",
            "--n-components",
            "7",
            "--no-secom",
            "--figure-format",
            "svg",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (seen["cfg"].alpha, seen["cfg"].n_components) == (0.02, 7)
    assert seen["with_secom"] is False
    assert seen["figures_dir"] == tmp_path.resolve() / "reports" / "figures"
    assert seen["fmt"] == "svg"
    assert (tmp_path / "reports" / "validation" / "tep_results.json").exists()
    assert "T2 0.010, SPE 0.020" in out


def test_validate_no_figures_keeps_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _fake_validation(monkeypatch)
    assert main(["validate", "tep", "--root", str(tmp_path), "--no-figures"]) == 0
    assert seen["cfg"] == secom.tep_validation.TEPValidationConfig()
    assert seen["with_secom"] is True
    assert "figures_dir" not in seen


@pytest.mark.parametrize("error", [FileNotFoundError("no tep data"), ValueError("bad shape")])
def test_validate_reports_input_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    _fake_validation(monkeypatch, error=error)
    assert main(["validate", "tep", "--root", str(tmp_path)]) == 2
    assert str(error) in capsys.readouterr().err


# monitor


class _FakeServer:
    url = "http://127.0.0.1:9999/"

    def __init__(self, interrupt: bool) -> None:
        self.interrupt = interrupt
        self.closed = False

    def serve_forever(self) -> None:
        if self.interrupt:
            raise KeyboardInterrupt

    def server_close(self) -> None:
        self.closed = True


def _fake_monitor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_error: Exception | None = None,
    serve_error: Exception | None = None,
    interrupt: bool = True,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    session = SimpleNamespace(runs=[1, 2, 3], design=SimpleNamespace(watch_sensor="S060"))

    def session_from_outputs(root: Path) -> SimpleNamespace:
        if session_error is not None:
            raise session_error
        return session

    def serve(sess: Any, **kwargs: Any) -> _FakeServer:
        if serve_error is not None:
            raise serve_error
        seen.update(kwargs)
        seen["server"] = _FakeServer(interrupt)
        return seen["server"]

    monkeypatch.setattr(secom.monitor, "session_from_outputs", session_from_outputs)
    monkeypatch.setattr(secom.monitor_server, "serve", serve)
    return seen


def test_monitor_serves_until_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _fake_monitor(monkeypatch)
    opened: list[str] = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", opened.append)
    code = main(
        ["monitor", "--root", str(tmp_path), "--port", "9999", "--speed", "5", "--open", "-v"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert seen["port"] == 9999 and seen["speed"] == 5.0 and seen["quiet"] is False
    assert seen["server"].closed
    assert opened == [_FakeServer.url]
    assert "3 runs, watching S060" in out


def test_monitor_returns_when_server_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _fake_monitor(monkeypatch, interrupt=False)
    assert main(["monitor", "--root", str(tmp_path)]) == 0
    assert seen["quiet"] is True
    assert seen["server"].closed


def test_monitor_rejects_non_positive_speed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["monitor", "--speed", "0"]) == 2
    assert "--speed must be positive" in capsys.readouterr().err


def test_monitor_without_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_monitor(monkeypatch, session_error=FileNotFoundError("run secom run first"))
    assert main(["monitor", "--root", str(tmp_path)]) == 2
    assert "run secom run first" in capsys.readouterr().err


def test_monitor_port_in_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_monitor(monkeypatch, serve_error=OSError("address in use"))
    assert main(["monitor", "--root", str(tmp_path), "--port", "1"]) == 2
    assert "cannot listen on 127.0.0.1:1" in capsys.readouterr().err


def test_monitor_export_writes_a_static_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _fake_monitor(monkeypatch)
    calls: list[tuple[Path, float]] = []

    def export_static(out: Path, design: Any, runs: Any, *, speed: float) -> list[Path]:
        calls.append((out, speed))
        return [out / "index.html"]

    monkeypatch.setattr("secom.monitor_export.export_static", export_static)
    out_dir = tmp_path / "static"
    assert main(["monitor", "--root", str(tmp_path), "--export", str(out_dir)]) == 0
    assert calls == [(out_dir, 20.0)]
    assert "server" not in seen  # nothing was served
    assert f"wrote {out_dir / 'index.html'}" in capsys.readouterr().out
