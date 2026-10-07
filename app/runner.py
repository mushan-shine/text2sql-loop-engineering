"""Run the loop from the console: a background thread + an event list the page polls.

Business logic only (no Streamlit): the page starts a ``LiveRun`` and re-renders its
``events`` every second. The run goes through the same code path as
``scripts/phase6.py`` (LoopController + evaluation.loop_run.run_arm), so a console
run and a CLI run of the same configuration are the same experiment.

Data the app needs is read from a bundle (``app/bundle/``, built by
``scripts/build_app_bundle.py``): schema catalog, few-shot examples, the dev set
(questions + MySQL gold rows for judging after the loop). The evaluation set is
read from Delta (benchmark.cases + benchmark.gold_results) and is locked unless
``SHT_ALLOW_EVAL=1`` (decision D2: evaluation runs only with a frozen configuration).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("SHT_BUNDLE", ROOT / "app" / "bundle"))
CATALOG = os.environ.get("SHT_CATALOG", "text2sql_loop")
MAX_CASES = int(os.environ.get("SHT_MAX_CASES_PER_RUN", "0"))  # 0 = no limit
MAX_REPAIRS = int(os.environ.get("SHT_MAX_REPAIRS", "4"))
MODELS = {"glm-4-flash": "zhipu", "deepseek-flash": "deepseek"}

log = logging.getLogger(__name__)


def eval_allowed() -> bool:
    return os.environ.get("SHT_ALLOW_EVAL", "").strip() in ("1", "true", "yes")


def load_dotenv() -> None:
    """Local runs: keys from .env. In Databricks Apps they come from secrets (app.yaml)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def model_available(model: str) -> bool:
    from agent.llm import PROVIDERS
    return bool(os.environ.get(PROVIDERS[MODELS[model]].key_env, "").strip())


# ------------------------------------------------------------------ data

@dataclass
class Bundle:
    catalog: Any
    examples: list
    devset: dict
    config: dict
    pool_path: Path | None = None
    _index: Any = None

    def example_index(self):
        """Dynamic few-shot pool (built lazily; None if the bundle predates it)."""
        if self._index is None and self.pool_path and self.pool_path.exists():
            import gzip
            from agent.examples import ExampleIndex
            with gzip.open(self.pool_path, "rt", encoding="utf-8") as f:
                self._index = ExampleIndex.from_rows(json.load(f))
        return self._index


def load_bundle(root: Path = BUNDLE) -> Bundle:
    from agent.generator import FewShotExample
    from agent.retriever import SchemaCatalog

    if not (root / "few_shot.json").exists():
        raise FileNotFoundError(f"app bundle missing at {root}; run: python scripts/build_app_bundle.py")
    catalog = SchemaCatalog.from_json((root / "schema_dw.json").read_text(encoding="utf-8"))
    examples = [FewShotExample(**e) for e in json.loads((root / "few_shot.json").read_text(encoding="utf-8"))]
    devset = json.loads((root / "devset.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return Bundle(catalog, examples, devset, config, root / "examples_pool.json.gz")


def dev_choices(bundle: Bundle) -> list[dict]:
    split = bundle.config["beaver"]["split"]
    return [{"case_id": f"{split}:{c['id']}", "question": c["question"], "category": c.get("category")}
            for c in bundle.devset["cases"]]


def eval_choices(conn) -> list[dict]:
    df = conn.query_df(f"SELECT c.case_id, c.question, c.category FROM `{CATALOG}`.`benchmark`.`cases` c "
                       f"JOIN `{CATALOG}`.`benchmark`.`gold_results` g ON c.case_id = g.case_id "
                       "WHERE g.evaluation_eligibility = 'PRIMARY' ORDER BY c.case_id")
    return df.to_dict("records")


def _cases_and_judges(bundle: Bundle, split: str, case_ids: list[str], conn) -> tuple[list, dict]:
    from benchmark.beaver.dataset import BeaverCase

    if split == "dev":
        from evaluation.devset import dev_cases, dev_judges
        s = bundle.config["beaver"]["split"]
        cases = [c for c in dev_cases(bundle.devset, s) if c.case_id in case_ids]
        judges = dev_judges(bundle.devset, s)
        return cases, {c.case_id: judges[c.case_id] for c in cases}
    if not eval_allowed():
        raise PermissionError("evaluation set is locked (decision D2); set SHT_ALLOW_EVAL=1 once the config is frozen")
    from evaluation.baseline import gold_judge
    ids = ", ".join("'" + c.replace("'", "") + "'" for c in case_ids)
    df = conn.query_df(f"SELECT c.case_id, c.question, c.db, c.category, g.gold_result "
                       f"FROM `{CATALOG}`.`benchmark`.`cases` c JOIN `{CATALOG}`.`benchmark`.`gold_results` g "
                       f"ON c.case_id = g.case_id WHERE g.evaluation_eligibility = 'PRIMARY' AND c.case_id IN ({ids})")
    cases, judges = [], {}
    for r in df.itertuples():
        cases.append(BeaverCase(case_id=r.case_id, split="eval", question=r.question, db=r.db, gold_sql="",
                                category=r.category))
        judges[r.case_id] = gold_judge(r.gold_result)
    return cases, judges


# ------------------------------------------------------------------ live run

@dataclass
class RunRequest:
    split: str                    # dev | eval
    case_ids: list[str]
    model: str = "glm-4-flash"
    strategy: str = "targeted"    # targeted | generic
    verifier: str = "self"        # self | oracle (upper bound)
    max_repairs: int = 1          # repair rounds per question; SQL attempts = max_repairs + 1
    few_shot: str = "static"      # static (fixed examples) | dynamic (similar solved questions, agent/examples.py)
    knowledge: bool = False       # add warehouse usage notes mined from solved queries (agent/knowledge.py)
    publish: bool = True


@dataclass
class LiveRun:
    request: RunRequest
    events: list[dict] = field(default_factory=list)
    status: str = "running"       # running | done | error
    error: str | None = None
    run_id: str | None = None
    summary: dict | None = None
    published: dict | None = None
    report: str | None = None      # markdown analysis report (app/report.py), set when the run finishes
    started: float = field(default_factory=time.time)
    finished: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def push(self, case_id: str, step: str, payload: dict) -> None:
        with self._lock:
            self.events.append({"t": round(time.time() - self.started, 1), "case_id": case_id, "step": step,
                                **payload})

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    @property
    def elapsed(self) -> float:
        return round((self.finished or time.time()) - self.started, 1)


def _cache_path() -> Path:
    """The shared phase-1 cache locally; a writable copy of the bundled cache in the app."""
    local = ROOT / "runs" / "phase1" / "llm_cache.jsonl"
    if local.parent.exists() and os.access(local.parent, os.W_OK) and not os.environ.get("DATABRICKS_APP_NAME"):
        return local
    tmp = Path(tempfile.gettempdir()) / "sht_llm_cache.jsonl"
    if not tmp.exists() and (BUNDLE / "llm_cache.jsonl").exists():
        shutil.copy(BUNDLE / "llm_cache.jsonl", tmp)
    return tmp


def _out_root() -> Path:
    local = ROOT / "runs" / "phase6"
    if os.environ.get("DATABRICKS_APP_NAME") or not os.access(ROOT, os.W_OK):
        return Path(tempfile.gettempdir()) / "sht_runs"
    return local


def build_controller(bundle: Bundle, inner: Any, conn: Any, strategy: str, max_repairs: int, few_shot: str,
                     knowledge: bool):
    """Same assembly as scripts/phase6.py: retriever, generator (+ dynamic few-shot / usage notes),
    executor, diagnoser, policy, repair context. Returns (controller, generator)."""
    from agent.generator import FewShotGenerator
    from agent.llm import CachingChatClient
    from agent.retriever import BM25TableRetriever
    from loop_engineer.controller import LoopConfig, LoopController
    from loop_engineer.diagnose import Diagnoser
    from loop_engineer.policy import Policy
    from skills.base import RepairContext

    cfg = bundle.config
    client = CachingChatClient(inner, _cache_path())
    fs = cfg.get("few_shot", {})
    index = bundle.example_index() if few_shot == "dynamic" else None
    if few_shot == "dynamic" and index is None:
        raise FileNotFoundError("examples_pool.json missing from the app bundle; rerun scripts/build_app_bundle.py")
    kb = None
    if knowledge:
        from agent.knowledge import load_knowledge
        kb = load_knowledge(BUNDLE / "kb.json")
        if kb is None:
            raise FileNotFoundError("kb.json missing from the app bundle; run scripts/build_knowledge.py and "
                                    "scripts/build_app_bundle.py")
    generator = FewShotGenerator(client, bundle.catalog, bundle.examples, index=index,
                                 k=int(fs.get("dynamic_k", 4)), max_extra_tables=int(fs.get("dynamic_max_extra_tables", 6)),
                                 knowledge=kb)
    controller = LoopController(BM25TableRetriever(bundle.catalog), generator, conn,
                                Diagnoser(bundle.catalog, client), Policy(),
                                RepairContext(bundle.catalog, client, bundle.examples),
                                LoopConfig(strategy=strategy, max_attempts=max_repairs + 1,
                                           top_k=int(cfg["retrieval"]["top_k"]),
                                           max_result_rows=int(cfg["databricks"]["max_result_rows"])))
    return controller, generator


def _execute(live: LiveRun, bundle: Bundle, connect) -> None:
    from agent.llm import UsageMeter, make_client
    from dbx.catalog import Layout
    from dbx.publish import publish_run
    from evaluation.loop_run import run_arm
    from loop_engineer.diagnose import DIAGNOSER_VERSION
    from loop_engineer.verifier import SELF_SIGNALS, VERIFIER_VERSION, OracleVerifier, SelfVerifier

    req, cfg = live.request, bundle.config
    lc = cfg["llm"]
    conn = connect()  # own warehouse connection per run thread
    try:
        cases, judges = _cases_and_judges(bundle, req.split, req.case_ids, conn)
        n, r = len(cases), req.max_repairs
        # per-click budget: per question 1 generation + per repair round (diagnosis LLM + repair LLM)
        calls = (1 + 2 * r) * n + 4
        meter = UsageMeter(max_calls=calls, max_tokens=30_000 * calls)
        inner = make_client(model=req.model, max_output_tokens=int(lc["max_output_tokens"]), meter=meter)
        controller, generator = build_controller(bundle, inner, conn, req.strategy, req.max_repairs, req.few_shot,
                                                 req.knowledge)
        verifier_for = ((lambda cid: SelfVerifier()) if req.verifier == "self"
                        else (lambda cid: OracleVerifier(judges[cid])))
        arm = f"console-{req.strategy}-{req.verifier}"
        meta = {"split": req.split, "strategy": req.strategy, "verifier": req.verifier,
                "upper_bound": req.verifier == "oracle", "policy": "targeted", "disabled": "", "source": "console",
                "self_signals": list(SELF_SIGNALS), "verifier_version": VERIFIER_VERSION, "model": inner.model, "provider": inner.provider,
                "prompt_version": generator.prompt_version, "few_shot": req.few_shot,
                "knowledge": "on" if req.knowledge else "off", "diagnoser": DIAGNOSER_VERSION,
                "max_attempts": req.max_repairs + 1, "max_repairs": req.max_repairs,
                "case_ids": [c.case_id for c in cases]}
        run_id, summary = run_arm(cases, judges, controller, verifier_for, arm, _out_root(), meta,
                                  on_event=live.push)
        summary["llm_usage"] = meter.snapshot()
        live.run_id, live.summary = run_id, summary
        from app.report import build_report
        live.report = build_report(live.snapshot(), asdict(req), summary, run_id, live.elapsed)
        (_out_root() / run_id / "report.md").write_text(live.report, encoding="utf-8")
        live.push("", "report", {"chars": len(live.report)})
        if req.publish:
            live.push("", "publish", {"state": "running"})
            live.published = publish_run(conn, Layout(CATALOG), _out_root() / run_id, "console", None)
            live.push("", "publish", {"state": "done", "run_id": run_id})
        live.status = "done"
    except Exception as e:  # surface every failure in the UI
        log.exception("console run failed")
        live.status, live.error = "error", f"{type(e).__name__}: {e}"
        live.push("", "error", {"message": live.error, "trace": traceback.format_exc()[-2000:]})
    finally:
        live.finished = time.time()
        try:
            conn.close()
        except Exception:
            pass


def start_run(request: RunRequest, bundle: Bundle, connect) -> LiveRun:
    if not request.case_ids:
        raise ValueError("pick at least one question")
    if MAX_CASES and len(request.case_ids) > MAX_CASES:
        raise ValueError(f"at most {MAX_CASES} questions per run")
    if request.model not in MODELS:
        raise ValueError(f"unknown model {request.model}")
    if not 1 <= request.max_repairs <= MAX_REPAIRS:
        raise ValueError(f"max_repairs must be 1..{MAX_REPAIRS}")
    live = LiveRun(request)
    live.push("", "queued", {"at": dt.datetime.now().strftime("%H:%M:%S"), "cases": len(request.case_ids)})
    threading.Thread(target=_execute, args=(live, bundle, connect), daemon=True, name="loop-run").start()
    return live
