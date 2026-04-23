"""Build Sphinx HTML and PDF (rinoh) for AQSparseBench."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only run the rinoh PDF builder",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only run the HTML builder",
    )
    args = parser.parse_args()
    do_html = not args.pdf_only
    do_pdf = not args.html_only

    cmds: list[list[str]] = []
    if do_html:
        cmds.append(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                str(docs),
                str(docs / "_build" / "html"),
            ]
        )
    if do_pdf:
        cmds.append(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "rinoh",
                str(docs),
                str(docs / "_build" / "rinoh"),
            ]
        )

    for cmd in cmds:
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    if do_pdf:
        out = docs / "_build" / "rinoh"
        pdfs = sorted(out.glob("*.pdf"))
        if pdfs:
            print("PDF:", pdfs[0], flush=True)
        else:
            print("Warning: no PDF found under", out, flush=True)
    if do_html:
        print("HTML:", docs / "_build" / "html" / "index.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
