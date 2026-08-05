from fda_device_rag.eval.questions import Question
from spot_check_transcripts import select_spot_check_questions


def _question(question_id, source_type):
    return Question(
        question_id=question_id, source_type=source_type, category=None,
        question=f"Question {question_id}?", gold={}, notes=None,
    )


def test_select_spot_check_questions_draws_per_stratum_count():
    questions = (
        [_question(f"recall-{i}", "recall") for i in range(13)]
        + [_question(f"maude-{i}", "maude") for i in range(13)]
        + [_question(f"guidance-{i}", "guidance") for i in range(12)]
        + [_question(f"ifu-{i}", "ifu") for i in range(12)]
    )

    selected = select_spot_check_questions(questions, seed="test-seed", per_stratum=4)

    assert len(selected) == 16
    by_type = {}
    for q in selected:
        by_type.setdefault(q.source_type, []).append(q)
    assert {st: len(qs) for st, qs in by_type.items()} == {
        "recall": 4, "maude": 4, "guidance": 4, "ifu": 4,
    }


def test_select_spot_check_questions_is_deterministic_for_a_given_seed():
    questions = [_question(f"recall-{i}", "recall") for i in range(13)]

    first = select_spot_check_questions(questions, seed="fixed-seed", per_stratum=4)
    second = select_spot_check_questions(questions, seed="fixed-seed", per_stratum=4)

    assert [q.question_id for q in first] == [q.question_id for q in second]


def test_select_spot_check_questions_strata_draw_independently():
    recall_questions = [_question(f"recall-{i}", "recall") for i in range(13)]
    maude_questions = [_question(f"maude-{i}", "maude") for i in range(13)]

    recall_only = select_spot_check_questions(recall_questions, seed="test-seed", per_stratum=4)
    combined = select_spot_check_questions(recall_questions + maude_questions, seed="test-seed", per_stratum=4)
    recall_from_combined = [q for q in combined if q.source_type == "recall"]

    assert [q.question_id for q in recall_only] == [q.question_id for q in recall_from_combined]
