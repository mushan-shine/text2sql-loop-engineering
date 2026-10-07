import json

from dbx.catalog import Layout
from dbx.publish import intervention_run_row, is_intervention_dir, publish_run


def _setup(tmp_path, summary_model=None):
    runs1 = tmp_path / "phase1"
    (runs1 / "baseline-x").mkdir(parents=True)
    (runs1 / "baseline-x" / "run_meta.json").write_text(json.dumps({"model": "glm-4-flash",
                                                                    "prompt_version": "baseline-v2"}))
    d = tmp_path / "phase3" / "intervention-baseline-x"
    d.mkdir(parents=True)
    summary = {"dev_run": "baseline-x", "failures": 27, "all_hints_fix_rate": "7/27",
               "llm_usage": {"input_tokens": 100, "output_tokens": 5}}
    if summary_model:
        summary["model"] = summary_model
    (d / "results.json").write_text(json.dumps({"summary": summary, "cases": [{"details": {"all": {"sql": "S"}}}]}))
    return runs1, d


def test_intervention_row_summary_only(tmp_path):
    runs1, d = _setup(tmp_path)
    assert is_intervention_dir(d)
    row = intervention_run_row(d, "model_probe", runs_root=runs1)
    assert (row["mode"], row["split"], row["model"]) == ("intervention", "dev", "glm-4-flash")  # from source run
    assert (row["correct"], row["cases"], row["tokens_total"]) == (7, 27, 105)
    assert row["prompt_version"] == "baseline-v2"
    assert '"sql"' not in row["summary_json"]  # gold-hinted SQL is never published


def test_summary_model_wins(tmp_path):
    runs1, d = _setup(tmp_path, summary_model="deepseek-flash")
    assert intervention_run_row(d, "model_probe", runs_root=runs1)["model"] == "deepseek-flash"


def test_publish_writes_only_evaluation_runs(tmp_path, monkeypatch):
    runs1, d = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (runs1).rename(tmp_path / "runs" / "phase1")
    written, sql = [], []
    monkeypatch.setattr("dbx.publish.table_exists", lambda *a: True)
    monkeypatch.setattr("dbx.publish.write_rows", lambda r, l, schema, name, rows, arrow: written.append((schema, name)))

    class Runner:
        def run(self, s):
            sql.append(s)

    out = publish_run(Runner(), Layout("c"), d, "model_probe", mlflow_experiment="ignored")
    assert written == [("evaluation", "runs")]
    assert all("traces" not in s for s in sql)
    assert out["all_hints_fixed"] == "7/27"
