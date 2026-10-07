"""Maintainer tool: export the prepared BEAVER data from a workspace into a course data package.

    python scripts/export_course_data.py --catalog <source catalog> --out course_data

Learners then load the package into their own workspace with scripts/load_course_data.py (no MySQL, no
step-1 replication). The package holds:

* tables/<schema>/<table>.parquet — the 97 ``dw`` tables and the ``benchmark`` tables (cases, table metadata,
  frozen gold results, compatibility and adaptation records, replication report);
* manifest.json — every table's columns with their exact Databricks types (``STRING COLLATE UTF8_LCASE``
  included: parquet does not keep collations), row counts, and the list of files;
* files/ — local artifacts the later steps read: data/beaver/<split>/ (questions + table metadata snapshot),
  runs/phase0/*.json (step-1 results for the report), runs/phase1/devset.json (dev set with MySQL gold rows).

BEAVER (https://huggingface.co/datasets/beaverbench/beaver-query, .../beaver-table) is MIT licensed; the
package README carries the attribution.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARK_TABLES = ("cases", "tables_meta", "gold_results", "gold_adaptations", "sql_compatibility",
                    "replication_report")
FILES = ["data/beaver/dw/dev.json", "data/beaver/dw/dev_tables.json", "runs/phase1/devset.json",
         "runs/phase0/01_environment.json", "runs/phase0/02_import.json", "runs/phase0/02b_replicate.json",
         "runs/phase0/03_compatibility.json", "runs/phase0/03_compatibility_run1_strict.json",
         "runs/phase0/04_gold_validation.json"]

README = """---
license: mit
pretty_name: text2sql-loop-engineering course data
---

# text2sql-loop-engineering course data

Data package for the course repository https://github.com/mushan-shine/text2sql-loop-engineering .
Load it into your own Databricks workspace from a notebook with `scripts/load_course_data.py`.

Contents: the BEAVER `dw` data warehouse (97 tables) and the benchmark tables prepared in step 1
(questions, table metadata, gold results validated against BEAVER's official MySQL engine), plus the
dev set and step-1 result files.

## Source and license

Derived from BEAVER (https://huggingface.co/datasets/beaverbench/beaver-query and
https://huggingface.co/datasets/beaverbench/beaver-table), released under the MIT License.
All credit for the benchmark goes to the BEAVER authors; see their dataset pages for the paper,
citation and full license text. This package only converts the data to Delta-ready parquet files.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="workspace catalog that already holds the step-1 data")
    ap.add_argument("--out", default="course_data")
    ap.add_argument("--source-root", default=str(ROOT), help="project folder holding data/ and runs/")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from execution.databricks_sql import DatabricksSqlExecutor

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    dbx = DatabricksSqlExecutor(catalog=args.catalog)
    manifest = {"source_catalog_note": "tables are created in the learner's own catalog", "tables": [], "files": []}
    targets = [("dw", r[1]) for r in dbx.run(f"SHOW TABLES IN `{args.catalog}`.`dw`")] + \
              [("benchmark", t) for t in BENCHMARK_TABLES]
    for schema, table in sorted(targets):
        fq = f"`{args.catalog}`.`{schema}`.`{table}`"
        cols = [(r[0], r[1]) for r in dbx.run(f"DESCRIBE TABLE {fq}") if r[0] and not r[0].startswith("#")]
        data = dbx.query_arrow(f"SELECT * FROM {fq}")
        # plain parquet: drop the connector's Spark field metadata, timestamps as UTC microseconds
        fields = [pa.field(f.name, pa.timestamp("us", tz="UTC") if pa.types.is_timestamp(f.type) else f.type)
                  for f in data.schema]
        data = pa.Table.from_arrays([c.cast(f.type) for c, f in zip(data.columns, fields)], schema=pa.schema(fields))
        path = out / "tables" / schema / f"{table}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data, path, version="2.4")
        manifest["tables"].append({"schema": schema, "table": table, "rows": data.num_rows,
                                   "columns": [{"name": n, "type": t} for n, t in cols],
                                   "file": str(path.relative_to(out)).replace("\\", "/")})
        print(f"{schema}.{table}: {data.num_rows} rows")
    dbx.close()

    src = Path(args.source_root)
    for rel in FILES:
        dst = out / "files" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / rel, dst)
        manifest["files"].append(rel)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"package: {out} — {len(manifest['tables'])} tables, {len(manifest['files'])} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
