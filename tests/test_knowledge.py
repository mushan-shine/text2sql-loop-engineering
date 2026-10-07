"""Outer-loop knowledge (agent/knowledge.py): counted from solved queries, rendered per question."""
from agent.generator import FewShotGenerator
from agent.knowledge import WarehouseKnowledge, knowledge_for
from agent.llm import LlmResponse
from agent.retriever import SchemaCatalog
from benchmark.beaver.dataset import AgentTask

CAT = SchemaCatalog.build("dw", [
    ("dept", "CODE", "STRING"), ("dept", "NAME", "STRING"),
    ("course_desc", "DEPT", "STRING"), ("course_desc", "LEVEL", "STRING"), ("course_desc", "TITLE", "STRING"),
    ("course_desc", "UNITS", "INT"),
    ("course_desc_hist", "DEPT", "STRING"), ("course_desc_hist", "LEVEL", "STRING"),
    ("course_desc_hist", "TITLE", "STRING"), ("course_desc_hist", "UNITS", "INT"),
], {})


def q(i, question, sql):
    return {"id": i, "question": question, "sql": sql}


QUERIES = [q(i, f"graduate level courses per department {i}",
             "SELECT d.NAME FROM DEPT d LEFT JOIN COURSE_DESC c ON c.DEPT = d.CODE WHERE c.LEVEL = 'G'")
           for i in range(8)] + \
          [q(100 + i, f"historical course titles {i}", "SELECT TITLE FROM COURSE_DESC_HIST") for i in range(6)] + \
          [q(200 + i, f"names of departments {i}", "SELECT NAME FROM DEPT") for i in range(6)] + \
          [q(999, "graduate level courses per department held out", "SELECT 1 FROM DEPT")]


def test_build_counts_tables_joins_groups_and_excludes_held_out():
    kb = WarehouseKnowledge.build(QUERIES, exclude_ids={"999"}, catalog=CAT)
    assert kb.n_queries == 20 and kb.table_freq["course_desc"] == 8
    e = kb.joins["course_desc|dept"]
    assert e["n"] == 8 and e["kinds"] == {"LEFT": 8} and "course_desc.DEPT = dept.CODE" in e["keys"]
    assert ["course_desc", "course_desc_hist"] in kb.groups           # look-alike tables (same columns)
    assert WarehouseKnowledge.from_json(kb.to_json()).joins == kb.joins


def test_notes_are_question_specific():
    kb = WarehouseKnowledge.build(QUERIES, exclude_ids={"999"}, catalog=CAT)
    notes = kb.notes_for("which graduate level courses does each department offer",
                         ["dept", "course_desc", "course_desc_hist"])
    assert "course_desc" in notes and "LEFT JOIN" in notes and "course_desc.DEPT = dept.CODE" in notes
    assert "Look-alike tables" in notes and "course_desc 100%" in notes
    assert kb.notes_for("zzz unrelated words", ["dept"]) == ""


def test_generator_adds_notes_and_versions_the_prompt(tmp_path):
    kb = WarehouseKnowledge.build(QUERIES, exclude_ids={"999"}, catalog=CAT)

    class Chat:
        model = "m"
        prompts = []

        def complete(self, prompt, system=None):
            self.prompts.append(prompt)
            return LlmResponse("```sql\nSELECT 1\n```", "m", 1, 1, 1)

    chat = Chat()
    gen = FewShotGenerator(chat, CAT, [], knowledge=kb)
    g = gen.generate(AgentTask("dw:1", "graduate level courses for each department", "dw"), ("dept", "course_desc"))
    assert "Warehouse usage notes" in chat.prompts[0] and g.notes and gen.prompt_version == "baseline-v2+kb"
    assert knowledge_for({"knowledge": {"mode": "off"}}, tmp_path) is None
