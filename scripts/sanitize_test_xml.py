"""Remove machine-specific details from a pytest JUnit XML file before publishing.

pytest records the machine's hostname and, for skipped tests, absolute file
paths (which contain the local user name). This script replaces the hostname
with "removed" and rewrites absolute project paths as project-relative paths.
Test names, counts, outcomes and timings are left unchanged.

Run from the project root after the tests:

    python -m pytest -q --junitxml=results/test_results.xml
    python scripts/sanitize_test_xml.py
"""

from __future__ import annotations

import argparse
import re
import sys

from _common import PROJECT_ROOT, resolve_input_path


def sanitize(text: str) -> str:
    """Return ``text`` with the hostname and absolute project paths removed."""
    text = re.sub(r'hostname="[^"]*"', 'hostname="removed"', text)
    root_variants = {str(PROJECT_ROOT), PROJECT_ROOT.as_posix()}
    for root in sorted(root_variants, key=len, reverse=True):
        for sep in ("\\", "/"):
            text = text.replace(root + sep, "")
        text = text.replace(root, ".")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strip hostname and absolute paths from a JUnit XML file.")
    parser.add_argument("--file", default="results/test_results.xml",
                        help="JUnit XML file to clean in place (default: results/test_results.xml).")
    args = parser.parse_args(argv)
    path = resolve_input_path(args.file)
    if not path.exists():
        print(f"error: {args.file} not found; run pytest with --junitxml first.", file=sys.stderr)
        return 1
    path.write_text(sanitize(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Sanitized {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
