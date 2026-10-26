"""Controlli di igiene: nessuna chiave nel codice, .env escluso da Git, .env.example senza valori."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]

# apiKey=<valore> oppure api_key = "<valore>" con un valore letterale non vuoto
HARDCODED = re.compile(r"""(?i)(api[_-]?key|apikey)\s*[=:]\s*["'][A-Za-z0-9_\-]{12,}["']""")


def test_no_hardcoded_api_keys_in_sources():
    offenders = [str(p.relative_to(ROOT)) for p in SOURCES if HARDCODED.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_env_is_gitignored_but_example_is_not():
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()}
    assert ".env" in lines
    assert "!.env.example" in lines


def test_env_example_has_no_secret_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.endswith("_API_KEY"):
            assert value.strip() == "", f"{name} deve essere vuoto in .env.example"


def test_sources_never_print_keys_directly():
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"print\([^)]*(api_key|_key)\b", text), f"possibile stampa di chiave in {path.name}"
