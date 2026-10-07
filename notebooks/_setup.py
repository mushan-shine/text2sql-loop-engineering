# Databricks notebook source
# MAGIC %md
# MAGIC ### 公共准备（被每一步的 notebook 用 `%run ./_setup` 引用）
# MAGIC
# MAGIC - 找到项目根目录，加入 `sys.path`，后面可以直接 `import` 项目代码；
# MAGIC - 从 Databricks secret 读取 LLM 的 key，放进环境变量（代码里不出现 key）；
# MAGIC - 提供 `run_script(...)`、`run_tests(...)`、`latest_run(...)`、`show_summary(...)` 几个小工具。

# COMMAND ----------

import json
import logging
import os
import runpy
import sys
from pathlib import Path

# Git folders do not accept __pycache__ directories: keep compiled files (and pytest's rewritten tests) in /tmp
sys.pycache_prefix = "/tmp/pycache"
os.environ["PYTHONPYCACHEPREFIX"] = "/tmp/pycache"

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

# the scripts log at INFO; keep the notebook output to our own messages
for _noisy in ("pyspark", "py4j", "mlflow", "databricks", "urllib3", "grpc"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SECRET_SCOPE = "text2sql-loop-engineering"
SECRET_KEYS = {"zhipu-key": "ZHIPUAI_API_KEY", "deepseek-key": "DEEPSEEK_API_KEY"}


def load_secrets() -> dict:
    """LLM keys from the secret scope into the environment. Missing keys are simply skipped."""
    found = {}
    for key, env in SECRET_KEYS.items():
        try:
            val = dbutils.secrets.get(SECRET_SCOPE, key)  # noqa: F821  (dbutils exists in notebooks)
        except Exception:
            val = ""
        if val and val != "not-configured":
            os.environ[env] = val
        found[env] = bool(val and val != "not-configured")
    return found


def run_script(script: str, *args: str) -> None:
    """Run scripts/<script> with command-line arguments, exactly like `python scripts/<script> ...`."""
    argv = sys.argv
    sys.argv = [script, *map(str, args)]
    try:
        runpy.run_path(str(REPO / "scripts" / script), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    finally:
        sys.argv = argv
        os.chdir(REPO)


def run_tests(*args: str) -> int:
    """pytest in its own Python process, from the project root (results printed below the cell).

    Not pytest.main(): inside the notebook's process pytest cannot import the project's packages from /Workspace,
    while a separate `python -m pytest` behaves exactly like a local run.
    """
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, args)],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return r.returncode


def latest_run(folder: str, prefix: str = "", mode: str | None = None) -> str:
    """Newest run directory under runs/<folder>/ (optionally only run_meta mode == ``mode``)."""
    dirs = sorted(p for p in (REPO / "runs" / folder).glob(f"{prefix}*") if p.is_dir())
    if mode:
        dirs = [d for d in dirs if (d / "run_meta.json").exists()
                and json.loads((d / "run_meta.json").read_text(encoding="utf-8")).get("mode") == mode]
    if not dirs:
        raise FileNotFoundError(f"no run under runs/{folder}/{prefix}*")
    return str(dirs[-1].relative_to(REPO)).replace("\\", "/")


def show_summary(run_dir: str) -> dict:
    """Key numbers of a run (runs/<...>/summary.json)."""
    s = json.loads((REPO / run_dir / "summary.json").read_text(encoding="utf-8"))
    keys = ("cases", "correct", "executable_rate", "mean_table_recall",                          # single generation
            "first_correct", "final_correct", "executable_first", "executable_final",            # inner loop
            "recovered", "harmed", "avg_attempts", "verifier_confusion", "tokens_mean", "tokens_total")
    picked = {k: s[k] for k in keys if k in s}
    print(json.dumps(picked, indent=1, ensure_ascii=False))
    return s


keys = load_secrets()
print(f"项目目录: {REPO}")
print("LLM key: " + ", ".join(f"{k} {'已配置' if v else '未配置'}" for k, v in keys.items()))
