import dataclasses
import random

from benchmark.beaver.dataset import AGENT_FIELDS, GOLD_FIELDS, AgentTask
from benchmark.beaver.loader import sample_official
from benchmark.beaver.report import decide_architecture


def test_agent_view_has_no_gold_fields(make_case):
    task = make_case().agent_view()
    names = {f.name for f in dataclasses.fields(AgentTask)}
    assert names == set(AGENT_FIELDS)
    assert not names & set(GOLD_FIELDS)
    assert not any(hasattr(task, g) for g in GOLD_FIELDS)


def test_gold_sql_preserved_verbatim(make_case):
    sql = "SELECT  `A` FROM T\nWHERE x = 'y' ;"
    assert make_case(sql).gold_sql == sql
    assert make_case(sql).to_row()["gold_sql"] == sql


def test_fingerprint_changes_with_content(make_case):
    assert make_case("SELECT 1").source_sha256 != make_case("SELECT 2").source_sha256
    assert make_case("SELECT 1").source_sha256 == make_case("SELECT 1").source_sha256


def test_sampling_equals_official_download_hf():
    entries = [{"id": i} for i in range(500)]
    random.seed(77)
    expected = random.sample(entries, 100)
    assert sample_official(entries, 100, 77) == expected
    assert len(sample_official(entries[:30], 100, 77)) == 30


def test_architecture_decision_thresholds():
    def compat(ok, total, ref_failed=0):
        return {"cases": total, "by_status": {"COMPATIBLE": ok, "REFERENCE_FAILED": ref_failed}}
    assert decide_architecture(compat(98, 100))[0] == "A"
    assert decide_architecture(compat(80, 100))[0] == "B"
    assert decide_architecture(compat(40, 100))[0] == "C"
    assert decide_architecture(compat(95, 100, ref_failed=5))[0] == "A"
