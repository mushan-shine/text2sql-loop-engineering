"""Deploy the console (Loop Debug Console + "Run loop" page) as a Databricks App.

    python scripts/build_app_bundle.py        # once, or after the dev set / few-shot changes
    python scripts/deploy_app.py              # upload code + bundle, create/update the app, deploy

Steps (idempotent, safe to re-run):
1. secrets   — scope ``text2sql-loop-engineering`` holds the LLM keys, read from the local .env.
               Values are never printed; only whether each key was set.
               In a Databricks notebook the keys already in the scope (notebooks/step00_setup) are kept.
2. upload    — the code the app imports + app/bundle/ + app.yaml / requirements.txt to
               /Workspace/Users/<you>/apps/<app name>. Tests, runs/, data/ and .env are never uploaded.
3. app       — created on first run with resources: SQL warehouse (CAN_USE) + the two secrets (READ).
4. grants    — the app's service principal may read the benchmark and write traces / evaluation
               rows in the project catalog (USE, SELECT, MODIFY, volume read/write for staging).
5. deploy    — deploys the uploaded folder and prints the app URL.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CODE_DIRS = ["agent", "loop_engineer", "skills", "execution", "dbx", "evaluation", "benchmark", "app"]
TOP_FILES = ["app.yaml", "requirements.txt", "config/phase1.yaml"]
SCOPE = "text2sql-loop-engineering"
SECRETS = {"zhipu-key": "ZHIPUAI_API_KEY", "deepseek-key": "DEEPSEEK_API_KEY"}


def read_env() -> dict[str, str]:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def files_to_upload() -> list[Path]:
    out = []
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.suffix == ".py" or (d == "app" and "bundle" in p.parts):
                out.append(p)
    out += [ROOT / f for f in TOP_FILES]
    missing = [p for p in out if not p.exists()]
    if missing or not (ROOT / "app/bundle/few_shot.json").exists():
        raise SystemExit(f"missing files {missing}; run scripts/build_app_bundle.py first")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="text2sql-loop-engineering")
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--catalog", default="text2sql_loop")
    ap.add_argument("--skip-secrets", action="store_true", help="keep the secrets already in the scope")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, ResourceAlreadyExists
    from databricks.sdk.service import apps
    from databricks.sdk.service.workspace import ImportFormat

    from dbx.runtime import in_databricks
    if in_databricks():  # notebook: the workspace authenticates us; keys are already in the scope (step 0)
        args.profile, args.skip_secrets = None, True
    w = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    me = w.current_user.me().user_name
    dest = f"/Workspace/Users/{me}/apps/{args.name}"

    # 1. secrets
    if args.skip_secrets:  # make sure the app's secret resources exist; never overwrite real keys
        try:
            w.secrets.create_scope(SCOPE)
        except ResourceAlreadyExists:
            pass
        have = {s.key for s in w.secrets.list_secrets(SCOPE)}
        for key in SECRETS:
            if key not in have:
                w.secrets.put_secret(SCOPE, key, string_value="not-configured")
                print(f"[secrets] {key}: missing -> placeholder (that model shows as unavailable)")
    else:
        env = read_env()
        try:
            w.secrets.create_scope(SCOPE)
            print(f"[secrets] created scope {SCOPE}")
        except ResourceAlreadyExists:
            pass
        for key, var in SECRETS.items():
            val = env.get(var, "") or os.environ.get(var, "")
            if val:
                w.secrets.put_secret(SCOPE, key, string_value=val)
                print(f"[secrets] {key}: set")
            else:
                w.secrets.put_secret(SCOPE, key, string_value="not-configured")
                print(f"[secrets] {key}: {var} not in .env -> placeholder (that model shows as unavailable)")

    # 2. upload
    files = files_to_upload()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        target = f"{dest}/{rel}"
        w.workspace.mkdirs(target.rsplit("/", 1)[0])
        w.workspace.upload(target, io.BytesIO(p.read_bytes()), format=ImportFormat.AUTO, overwrite=True)
    print(f"[upload] {len(files)} files -> {dest}")

    # 3. app
    wh_id = _warehouse_id(w)
    resources = [
        apps.AppResource(name="sql-warehouse", sql_warehouse=apps.AppResourceSqlWarehouse(
            id=wh_id, permission=apps.AppResourceSqlWarehouseSqlWarehousePermission.CAN_USE)),
        *[apps.AppResource(name=key, secret=apps.AppResourceSecret(
            scope=SCOPE, key=key, permission=apps.AppResourceSecretSecretPermission.READ)) for key in SECRETS],
    ]
    try:
        app = w.apps.get(args.name)
        print(f"[app] exists: {args.name} ({app.compute_status.state if app.compute_status else '?'})")
        w.apps.update(args.name, apps.App(name=args.name, resources=resources,
                                          description="Self-healing Text-to-SQL: loop console with live runs"))
    except NotFound:
        print(f"[app] creating {args.name} (this starts its compute; takes a few minutes)")
        app = w.apps.create_and_wait(apps.App(name=args.name, resources=resources,
                                              description="Self-healing Text-to-SQL: loop console with live runs"))
    app = w.apps.get(args.name)
    if app.compute_status and str(app.compute_status.state).endswith(("STOPPED", "STOPPING")):
        print("[app] starting compute")
        w.apps.start_and_wait(args.name)
        app = w.apps.get(args.name)

    # 4. grants for the app's service principal
    sp = app.service_principal_client_id
    from execution.databricks_sql import DatabricksSqlExecutor
    dbx = DatabricksSqlExecutor(catalog=args.catalog, profile=args.profile)
    for stmt in (f"GRANT USE CATALOG ON CATALOG `{args.catalog}` TO `{sp}`",
                 f"GRANT USE SCHEMA, SELECT, MODIFY, READ VOLUME, WRITE VOLUME, CREATE TABLE "
                 f"ON CATALOG `{args.catalog}` TO `{sp}`"):
        dbx.run(stmt)
    dbx.close()
    print(f"[grants] catalog {args.catalog} -> app service principal {sp}")

    # 5. deploy
    print("[deploy] deploying (a few minutes)")
    dep = w.apps.deploy_and_wait(args.name, apps.AppDeployment(source_code_path=dest))
    print(f"[deploy] {dep.status.state if dep.status else '?'}: {dep.status.message if dep.status else ''}")
    print(f"[done] {w.apps.get(args.name).url}")


def _warehouse_id(w) -> str:
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path:
        return http_path.rstrip("/").rsplit("/", 1)[-1]
    whs = list(w.warehouses.list())
    running = [x for x in whs if str(x.state).endswith("RUNNING")]
    return (running or whs)[0].id


if __name__ == "__main__":
    main()
