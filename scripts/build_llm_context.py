#!/usr/bin/env python3
"""Generate LLM-friendly context files for the repository.

Produces two artifacts, so a researcher can drop the whole project into an LLM:

  * ``llms.txt``       -- a short, curated index (llmstxt.org convention):
                          one-line descriptions + links to the important files.
  * ``llms-full.txt``  -- every source/doc/result file concatenated into a
                          single pasteable document, with clear file separators
                          and a manifest header (file count, size, ~token count).

Deterministic output (stable ordering) so CI diffs are meaningful.

Usage:
    python scripts/build_llm_context.py            # regenerate both files
    python scripts/build_llm_context.py --check     # exit 1 if out of date
"""

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Files included in the full bundle, in reading order. Globs are sorted.
INCLUDE_GLOBS = [
    "README.md",
    "NUTSHELL.md",
    "RELEVANCE.md",
    "rules.md",
    "NOTATION.md",
    "30SECONDS.md",
    "docs/*.md",
    "pyproject.toml",
    "CITATION.cff",
    "conftest.py",
    "collapse3/*.py",
    "experiments/*.py",
    "probes/*.py",
    "tests/*.py",
    "scripts/*.py",
    "results/*_latest.json",
]

# Short descriptions for the curated index; falls back to a module's docstring.
DESCRIPTIONS = {
    "README.md": "Project front page: thesis, headline result, quickstart.",
    "NUTSHELL.md": "Brief plain-language overview of the game and why it matters.",
    "RELEVANCE.md": "Why a solved toy game is about your metrics, not your model: the transfer test.",
    "rules.md": "Complete Collapse3 game rules.",
    "NOTATION.md": "Plain-text board notation (produced by collapse3/render.py).",
    "30SECONDS.md": "30-second explainer: winning shapes and the gravity collapse.",
    "docs/FINDINGS.md": "Full research findings and their significance.",
    "docs/NEURAL_EXHIBIT.md": "Finding 16 in full: a net that generalizes is still certifiably force-losable (torch exhibit).",
    "docs/EVALUATION_EQUIVALENCE.md": "Finding 17 in full: what passing an evaluation rules out; the strongest player is not the strongest tester (exact, pure-Python).",
    "pyproject.toml": "Package metadata, dependencies, pytest config.",
}

INDEX_SECTIONS = [
    ("Start here", ["NUTSHELL.md", "README.md", "RELEVANCE.md", "rules.md",
                    "NOTATION.md", "docs/FINDINGS.md", "docs/NEURAL_EXHIBIT.md",
                    "docs/EVALUATION_EQUIVALENCE.md"]),
    ("Engine & tools", ["collapse3/game.py", "collapse3/solver.py",
                        "collapse3/enumeration.py", "collapse3/oracle.py",
                        "collapse3/agents.py", "collapse3/learning.py",
                        "collapse3/aliasing.py", "collapse3/metrics.py"]),
    # Experiments are discovered from the glob at build time (see below) so the
    # index cannot drift behind the bundle when new experiments land.
    ("Experiments", sorted(
        str(p.relative_to(REPO)) for p in REPO.glob("experiments/*.py")
        if p.name not in ("_provenance.py", "__init__.py")
    )),
]


def describe(rel: str) -> str:
    if rel in DESCRIPTIONS:
        return DESCRIPTIONS[rel]
    path = REPO / rel
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
            if doc:
                return doc.strip().splitlines()[0]
        except (SyntaxError, OSError):
            pass
    return ""


def collect_files() -> list[Path]:
    seen, files = set(), []
    for pattern in INCLUDE_GLOBS:
        for p in sorted(REPO.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def build_index() -> str:
    lines = [
        "# Collapse3",
        "",
        "> Exact-solver-graded study of competence vs. performance in game AI: "
        "using a fully solved 3x3x3 game as an oracle to measure value-based "
        "regret independent of any opponent.",
        "",
    ]
    for title, rels in INDEX_SECTIONS:
        lines.append(f"## {title}")
        for rel in rels:
            if (REPO / rel).exists():
                desc = describe(rel)
                lines.append(f"- [{rel}]({rel})" + (f": {desc}" if desc else ""))
        lines.append("")
    lines.append("## Full context")
    lines.append("- [llms-full.txt](llms-full.txt): entire repository in one file "
                 "(paste into an LLM to ask about the research).")
    lines.append("")
    return "\n".join(lines)


def build_full(files: list[Path]) -> str:
    bodies = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        bar = "=" * 78
        bodies.append(f"{bar}\nFILE: {rel}\n{bar}\n{text}\n")
    joined = "\n".join(bodies)

    chars = len(joined)
    manifest = [
        "# Collapse3 -- full repository context bundle",
        "# Generated by scripts/build_llm_context.py -- DO NOT EDIT BY HAND.",
        f"# {len(files)} files, {chars:,} chars, ~{chars // 4:,} tokens (rough).",
        "#",
        "# Files, in order:",
    ]
    manifest += [f"#   {p.relative_to(REPO).as_posix()}" for p in files]
    manifest.append("")
    return "\n".join(manifest) + "\n" + joined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if generated files are stale (for CI)")
    args = ap.parse_args()

    files = collect_files()
    index = build_index()
    full = build_full(files)
    targets = {REPO / "llms.txt": index, REPO / "llms-full.txt": full}

    if args.check:
        stale = [p.name for p, content in targets.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != content]
        if stale:
            print(f"STALE: {', '.join(stale)} -- run scripts/build_llm_context.py")
            return 1
        print("LLM context files are up to date.")
        return 0

    for p, content in targets.items():
        p.write_text(content, encoding="utf-8")
    print(f"Wrote llms.txt and llms-full.txt "
          f"({len(files)} files, {len(full):,} chars, ~{len(full)//4:,} tokens).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
