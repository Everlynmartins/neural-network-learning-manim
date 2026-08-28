from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "neural_network_training.ipynb"
README = ROOT / "README.md"
EXPECTED_ASSETS = (
    ROOT / "assets" / "images" / "01-real-arithmetic.png",
    ROOT / "assets" / "images" / "02-relu-activation.png",
    ROOT / "assets" / "images" / "03-output-aggregation.png",
    ROOT / "assets" / "images" / "04-final-boundary.png",
    ROOT / "assets" / "images" / "05-backpropagation.png",
    ROOT / "assets" / "videos" / "neural-network-learning-1080x1350-2x.mp4",
)
FORBIDDEN_TERMS = ("chat" + "gpt", "open" + "ai", "co" + "dex")
TEXT_SUFFIXES = {".md", ".py", ".ps1", ".txt", ".yml", ".yaml", ".ipynb", ".cff"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def validate_notebook(execute: bool) -> tuple[int, int]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    cells = notebook.get("cells", [])
    markdown_count = sum(cell.get("cell_type") == "markdown" for cell in cells)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    assert markdown_count >= 8
    assert len(code_cells) >= 8

    namespace: dict[str, object] = {"__name__": "__notebook_validation__"}
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"notebook-cell-{index}")
        if execute:
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
    return markdown_count, len(code_cells)


def validate_assets() -> None:
    for asset in EXPECTED_ASSETS:
        assert asset.is_file(), f"Ativo ausente: {asset.relative_to(ROOT)}"
        assert asset.stat().st_size > 0, f"Ativo vazio: {asset.relative_to(ROOT)}"

    video = EXPECTED_ASSETS[-1]
    assert video.stat().st_size < 100 * 1024 * 1024

    readme = README.read_text(encoding="utf-8")
    for asset in EXPECTED_ASSETS:
        relative = asset.relative_to(ROOT).as_posix()
        assert relative in readme, f"README sem referência a {relative}"


def validate_public_text() -> None:
    windows_user_path = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)
    for path in tracked_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        lowered = text.lower()
        for term in FORBIDDEN_TERMS:
            assert term not in lowered, f"Termo proibido em {path.relative_to(ROOT)}"
        assert not windows_user_path.search(text), f"Caminho local em {path.relative_to(ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-notebook", action="store_true")
    args = parser.parse_args()

    validate_assets()
    markdown_count, code_count = validate_notebook(args.execute_notebook)
    validate_public_text()
    print(f"notebook_markdown_cells={markdown_count}")
    print(f"notebook_code_cells={code_count}")
    print(f"notebook_executed={args.execute_notebook}")
    print("public_release_validation=OK")


if __name__ == "__main__":
    main()
