"""LLM judge: robust JSON parsing and a prompt that carries the warehouse conventions."""
from agent.llm import LlmResponse
from loop_engineer.judge import CONVENTIONS, LlmJudge, parse_judgement
from tests.test_phase6 import CAT


class Chat:
    model = "m"

    def __init__(self, text):
        self.text, self.prompts = text, []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return LlmResponse(self.text, "m", 50, 10, 5)


def test_parse_judgement():
    assert parse_judgement('Sure: {"verdict": "Wrong", "confidence": 0.9, "problems": ["missing filter"]}') == \
        ("wrong", 0.9, ("missing filter",))
    assert parse_judgement('{"verdict": "correct", "confidence": 7}')[1] == 1.0   # clamped
    assert parse_judgement('{"verdict": "maybe"}') is None
    assert parse_judgement("no json here") is None


def test_judge_prompt_and_unparseable_answer():
    chat = Chat("I think it is fine")
    j = LlmJudge(chat, CAT).judge("average per department", "SELECT d.DEPARTMENT_NAME FROM SIS_DEPARTMENT d",
                                  [("Math",)], 1)
    assert j.verdict == "unparseable" and j.tokens == 60
    p = chat.prompts[0]
    assert CONVENTIONS in p and "TABLE dw.sis_department" in p and '"Math"' in p
