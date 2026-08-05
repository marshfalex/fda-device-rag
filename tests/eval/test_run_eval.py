from run_eval import _score_all

from fda_device_rag.eval.gold_resolution import GoldResolutionError


class _FakeQuestion:
    def __init__(self, question_id):
        self.question_id = question_id


def test_score_all_excludes_resolution_errors_from_per_question():
    questions = [_FakeQuestion("q1"), _FakeQuestion("q2"), _FakeQuestion("q3")]

    def scorer(question):
        if question.question_id == "q2":
            raise GoldResolutionError("no chunk matched the gold locator")
        return {"question_id": question.question_id, "rank": 1}

    per_question, resolution_errors = _score_all(questions, scorer)

    assert [pq["question_id"] for pq in per_question] == ["q1", "q3"]
    assert all(pq["rank"] is not None for pq in per_question)
    assert [e["question_id"] for e in resolution_errors] == ["q2"]


def test_score_all_no_errors_returns_empty_resolution_errors():
    questions = [_FakeQuestion("q1"), _FakeQuestion("q2")]

    per_question, resolution_errors = _score_all(questions, lambda q: {"question_id": q.question_id})

    assert len(per_question) == 2
    assert resolution_errors == []
