"""Dynamic few-shot (agent/examples.py): exclusions, ranking, and the schema the generator shows."""
from agent.examples import ExampleIndex, build_generator_index
from agent.generator import FewShotGenerator
from agent.llm import LlmResponse
from benchmark.beaver.dataset import AgentTask
from tests.test_phase6 import CAT

QUERIES = [
    {"id": 1, "question": "average enrollment per department", "sql":
        "SELECT d.DEPARTMENT_NAME, AVG(s.NUM_ENROLLED) FROM SIS_DEPARTMENT d JOIN SUBJECT_OFFERED s "
        "ON d.DEPARTMENT_CODE = s.DEPARTMENT_CODE GROUP BY d.DEPARTMENT_NAME"},
    {"id": 2, "question": "list department names", "sql": "SELECT DEPARTMENT_NAME FROM SIS_DEPARTMENT"},
    {"id": 3, "question": "average enrollment per department (eval)", "sql": "SELECT 1"},
    {"id": 4, "question": "broken", "sql": "SELECT FROM WHERE ((("},
    {"id": 5, "question": "no sql"},
]


class Chat:
    model = "m"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse("```sql\nSELECT 1\n```", "m", 10, 5, 1)


def test_pool_excludes_eval_dev_unparseable_and_missing_sql():
    idx = ExampleIndex.build(QUERIES, exclude_ids={"3"})
    assert [e.example.source_id for e in idx.entries] == ["1", "2"]
    assert idx.entries[0].tables == ("sis_department", "subject_offered")
    assert build_generator_index(QUERIES, {"3"}, {"2"}, {"mode": "static"}) is None
    dyn = build_generator_index(QUERIES, {"3"}, {"2"}, {"mode": "dynamic"})
    assert [e.example.source_id for e in dyn.entries] == ["1"]
    assert [e.example.source_id for e in ExampleIndex.from_rows(dyn.to_rows()).entries] == ["1"]


def test_most_similar_first_and_example_tables_join_the_schema():
    idx = ExampleIndex.build(QUERIES, exclude_ids={"3"})
    assert idx.top("average enrollment for each department", 1)[0].example.source_id == "1"
    chat = Chat()
    gen = FewShotGenerator(chat, CAT, [], index=idx, k=1)
    g = gen.generate(AgentTask("dw:9", "average enrollment for each department", "dw"), ("sis_department",))
    assert g.example_ids == ("1",) and g.schema_tables == ("sis_department", "subject_offered")
    assert "TABLE dw.subject_offered" in chat.prompts[0] and "AVG(s.NUM_ENROLLED)" in chat.prompts[0]
    assert gen.prompt_version == "baseline-v3.1-dynfs"
    assert FewShotGenerator(chat, CAT, []).prompt_version == "baseline-v2.1"
