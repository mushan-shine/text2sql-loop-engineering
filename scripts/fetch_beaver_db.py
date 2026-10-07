"""Download BEAVER's anonymized MySQL dump (beaver_db.zip, ~262 MB, gated on HF)
and print the commands to load it into the local reference MySQL server.

Prerequisite: accept the terms on https://huggingface.co/datasets/beaverbench/beaver-table
and run `hf auth login` once.

The archive was built on macOS: `__MACOSX/`, `._*` and `.DS_Store` entries are
resource-fork junk and are skipped. Each dump starts with `CREATE DATABASE <db>; USE <db>;`.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "beaver_db"
MYSQL = r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" # 替换为自己本地执行环境的MySQL路径


def is_junk(name: str) -> bool:
    parts = Path(name).parts
    return "__MACOSX" in parts or Path(name).name.startswith("._") or Path(name).name == ".DS_Store"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", default="dw", help="comma-separated dumps to print import commands for (dw,neutron,nova)")
    args = ap.parse_args()
    wanted = {d.strip() for d in args.dbs.split(",") if d.strip()}

    from huggingface_hub import hf_hub_download

    zpath = Path(hf_hub_download("beaverbench/beaver-table", "beaver_db.zip", repo_type="dataset"))
    print(f"downloaded {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DEST, members=[m for m in z.namelist() if not is_junk(m)])
    dumps = sorted(p for p in DEST.rglob("*.sql") if not is_junk(str(p.relative_to(DEST))))
    if not dumps:
        sys.exit(f"no .sql files found under {DEST}; inspect the archive layout")
    print("\nDumps:")
    for d in dumps:
        print(f"  {d.stem:10s} {d.stat().st_size / 1e6:9.1f} MB  {d}")
    print("\nImport in a cmd window (you will be prompted for the password):")
    for d in dumps:
        if d.stem in wanted:
            print(f'  "{MYSQL}" -u root -p --default-character-set=utf8mb4 < "{d}"')


if __name__ == "__main__":
    main()
