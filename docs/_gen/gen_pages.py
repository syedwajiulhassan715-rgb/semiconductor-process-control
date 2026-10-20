"""Build-time page generation for the MkDocs site (run by mkdocs-gen-files).

Nothing here imports ``secom``: the stage registry and config sections are read
with ``ast``, and the API pages are rendered by mkdocstrings from source. So the
site builds without numpy/pandas installed, and anything merged later (a new
module, stage or config section) appears without touching ``mkdocs.yml``.

Repository documents (the engineering report, the decision log, the Power BI
guide) are copied in rather than duplicated. Their relative links are written
for GitHub, so they are rewritten here:

* a link to another document on the site points at that page;
* a link to an image is copied into ``assets/`` and pointed there;
* a link to anything else in the repo (code, data, CI config) goes to GitHub.

A relative link whose target does not exist is reported as a warning, which
``mkdocs build --strict`` turns into a failure.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from pathlib import Path

import mkdocs_gen_files

log = logging.getLogger("mkdocs.plugins.secom_gen_pages")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PKG = SRC / "secom"
REPO_URL = "https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control"
BRANCH = "main"
TEMPLATES = Path(__file__).resolve().parent

#: Repository document -> page on the site.
DOCUMENTS = {
    "reports/ENGINEERING_REPORT.md": "report.md",
    "DECISIONS.md": "decisions.md",
    "powerbi/README.md": "powerbi.md",
}
ASSET_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"}

_MD_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\s+\"[^\"]*\")?\)")
_HTML_ATTR = re.compile(r"""(\b(?:src|href)=")([^"]+)(")""")


# ---------------------------------------------------------------- documents


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#", "/"))


def _rewrite_target(target: str, source: str, page: str) -> str:
    """Rewrite one relative link found in repo file ``source``, now at ``page``."""
    if _is_external(target):
        return target
    path, _, anchor = target.partition("#")
    anchor = f"#{anchor}" if anchor else ""
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    page_dir = posixpath.dirname(page)

    if repo_path in DOCUMENTS:
        return posixpath.relpath(DOCUMENTS[repo_path], page_dir or ".") + anchor

    on_disk = ROOT / repo_path
    if not on_disk.exists():
        log.warning("%s links to %s, which does not exist", source, repo_path)
        return target

    if on_disk.is_file() and on_disk.suffix.lower() in ASSET_SUFFIXES:
        asset = f"assets/{repo_path}"
        with mkdocs_gen_files.open(asset, "wb") as fh:
            fh.write(on_disk.read_bytes())
        return posixpath.relpath(asset, page_dir or ".")

    kind = "tree" if on_disk.is_dir() else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{repo_path.rstrip('/')}{anchor}"


def _rewrite(text: str, source: str, page: str) -> str:
    def md(m: re.Match[str]) -> str:
        title = m.group(3) or ""
        return f"{m.group(1)}{_rewrite_target(m.group(2), source, page)}{title})"

    def html(m: re.Match[str]) -> str:
        return f"{m.group(1)}{_rewrite_target(m.group(2), source, page)}{m.group(3)}"

    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            line = _HTML_ATTR.sub(html, _MD_LINK.sub(md, line))
        out.append(line)
    return "".join(out)


def generate_documents() -> None:
    for source, page in DOCUMENTS.items():
        text = (ROOT / source).read_text(encoding="utf-8")
        with mkdocs_gen_files.open(page, "w", encoding="utf-8") as fh:
            fh.write(_rewrite(text, source, page))
        mkdocs_gen_files.set_edit_path(page, f"../{source}")


# ------------------------------------------------------------- static parse


def _literal(node: ast.AST | None) -> object:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def _module_assigns(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            out[node.target.id] = node.value
    return out


def _stage_registry() -> list[dict[str, object]]:
    init = PKG / "pipeline" / "__init__.py"
    names = _module_assigns(ast.parse(init.read_text(encoding="utf-8")))
    stages_node = names.get("STAGES")
    if not isinstance(stages_node, ast.List):
        raise RuntimeError("could not find the STAGES list in secom.pipeline")
    stages = []
    for elt in stages_node.elts:
        mod = elt.id if isinstance(elt, ast.Name) else ast.unparse(elt)
        path = PKG / "pipeline" / "stages" / f"{mod}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values = _module_assigns(tree)
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        stages.append(
            {
                "module": mod,
                "name": _literal(values.get("NAME")) or mod,
                "requires": tuple(_literal(values.get("REQUIRES")) or ()),
                "produces": tuple(_literal(values.get("PRODUCES")) or ()),
                "summary": doc[0] if doc else "",
            }
        )
    return stages


def _field_comment(lines: list[str], lineno: int) -> str:
    """The ``#:`` comment block directly above a dataclass field."""
    out: list[str] = []
    i = lineno - 2
    while i >= 0 and lines[i].strip().startswith("#:"):
        out.insert(0, lines[i].strip()[2:].strip())
        i -= 1
    return " ".join(out)


def _registered_sections() -> list[dict[str, object]]:
    sections = []
    for path in sorted(PKG.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "register_section" not in source:
            continue
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            key = None
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "register_section"
                    and dec.args
                ):
                    key = _literal(dec.args[0])
            if key is None:
                continue
            fields = [
                {
                    "name": stmt.target.id,
                    "type": ast.unparse(stmt.annotation),
                    "default": _literal(stmt.value),
                    "doc": _field_comment(lines, stmt.lineno),
                }
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            module = ".".join(path.relative_to(SRC).with_suffix("").parts)
            sections.append({"key": key, "cls": node.name, "module": module, "fields": fields})
    return sorted(sections, key=lambda s: str(s["key"]))


# ------------------------------------------------------------ guide pages


def _yaml_value(value: object) -> str:
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def generate_configuration() -> None:
    blocks = []
    for sec in _registered_sections():
        rows = "\n".join(
            f"| `{f['name']}` | `{_cell(str(f['type']))}` | `{_cell(_yaml_value(f['default']))}` "
            f"| {_cell(str(f['doc']))} |"
            for f in sec["fields"]  # type: ignore[union-attr]
        )
        yaml = "\n".join(
            f"  {f['name']}: {_yaml_value(f['default'])}"
            for f in sec["fields"]  # type: ignore[union-attr]
        )
        blocks.append(
            f"### `{sec['key']}`\n\n"
            f"Registered by `{sec['cls']}` in [`{sec['module']}`][{sec['module']}].\n\n"
            "| Key | Type | Default | Meaning |\n|---|---|---|---|\n"
            f"{rows}\n\n"
            f'```yaml title="defaults"\n{sec["key"]}:\n{yaml}\n```\n'
        )
    sections = "\n".join(blocks) or "_No stage registers its own section yet._\n"
    default_yaml = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    text = (TEMPLATES / "configuration.md").read_text(encoding="utf-8")
    text = text.replace("{{DEFAULT_YAML}}", default_yaml.rstrip())
    text = text.replace("{{SECTIONS}}", sections.rstrip())
    with mkdocs_gen_files.open("configuration.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    mkdocs_gen_files.set_edit_path("configuration.md", "../config/default.yaml")


def generate_architecture() -> None:
    stages = _stage_registry()
    producer = {key: s["name"] for s in stages for key in s["produces"]}  # type: ignore[attr-defined]

    rows = "\n".join(
        f"| {i} | `{s['name']}` | {_cell(str(s['summary']))} | "
        f"{', '.join(f'`{k}`' for k in s['requires']) or '-'} | "  # type: ignore[attr-defined]
        f"{', '.join(f'`{k}`' for k in s['produces']) or '-'} |"  # type: ignore[attr-defined]
        for i, s in enumerate(stages, start=1)
    )

    edges: dict[tuple[str, str], list[str]] = {}
    for s in stages:
        for key in s["requires"]:  # type: ignore[attr-defined]
            src = producer.get(key)
            if src is not None:
                edges.setdefault((str(src), str(s["name"])), []).append(key)
    nodes = "\n".join(f'    {s["name"]}["{s["name"]}"]' for s in stages)
    links = "\n".join(
        f'    {a} -->|"{", ".join(keys[:3])}{", ..." if len(keys) > 3 else ""}"| {b}'
        for (a, b), keys in edges.items()
    )
    mermaid = f"```mermaid\nflowchart TD\n{nodes}\n{links}\n```"

    text = (TEMPLATES / "architecture.md").read_text(encoding="utf-8")
    text = text.replace("{{STAGE_TABLE}}", rows).replace("{{STAGE_DIAGRAM}}", mermaid)
    with mkdocs_gen_files.open("architecture.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    mkdocs_gen_files.set_edit_path("architecture.md", "../src/secom/pipeline/__init__.py")


# ----------------------------------------------------------- API reference


def generate_reference() -> None:
    nav = mkdocs_gen_files.Nav()
    for path in sorted(PKG.rglob("*.py")):
        module_path = path.relative_to(SRC).with_suffix("")
        parts = tuple(module_path.parts)
        if parts[-1] == "__main__":
            continue
        doc_path = Path("reference", *parts).with_suffix(".md")
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = Path("reference", *parts, "index.md")
        nav[parts] = doc_path.relative_to("reference").as_posix()
        with mkdocs_gen_files.open(doc_path.as_posix(), "w", encoding="utf-8") as fh:
            fh.write(f"::: {'.'.join(parts)}\n")
        source = path.relative_to(ROOT).as_posix()
        mkdocs_gen_files.set_edit_path(doc_path.as_posix(), f"../{source}")
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w", encoding="utf-8") as fh:
        fh.writelines(nav.build_literate_nav())


generate_documents()
generate_configuration()
generate_architecture()
generate_reference()
