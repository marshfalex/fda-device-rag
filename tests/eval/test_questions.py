import json

import pytest

from fda_device_rag.eval.questions import Question, QuestionsFileError, load_questions


def _write(tmp_path, data):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data))
    return path


def test_load_questions_parses_valid_file(tmp_path):
    path = _write(tmp_path, {
        "version": 1,
        "frozen_date": "2026-08-03",
        "questions": [
            {
                "question_id": "recall-01",
                "source_type": "recall",
                "category": "Process control",
                "question": "Why was the XYZ pump recalled?",
                "gold": {"record_id": "Z-0001-2024"},
                "notes": None,
            },
            {
                "question_id": "guidance-01",
                "source_type": "guidance",
                "question": "What software validation activities does FDA recommend?",
                "gold": {"document_title": "188844", "section_name": "Risk-Based Analysis", "instance_ordinal": 1},
            },
        ],
    })

    questions = load_questions(path)

    assert len(questions) == 2
    assert questions[0] == Question(
        question_id="recall-01", source_type="recall", category="Process control",
        question="Why was the XYZ pump recalled?", gold={"record_id": "Z-0001-2024"}, notes=None,
    )
    # category/notes default to None when absent from the raw JSON
    assert questions[1].category is None
    assert questions[1].notes is None


def test_load_questions_raises_on_missing_questions_key(tmp_path):
    path = _write(tmp_path, {"version": 1})

    with pytest.raises(QuestionsFileError, match="missing top-level 'questions' key"):
        load_questions(path)


def test_load_questions_raises_on_missing_required_field(tmp_path):
    path = _write(tmp_path, {"questions": [{"question_id": "q1", "source_type": "recall", "gold": {}}]})

    with pytest.raises(QuestionsFileError, match="question"):
        load_questions(path)


def test_load_questions_raises_on_invalid_source_type(tmp_path):
    path = _write(tmp_path, {"questions": [{
        "question_id": "q1", "source_type": "bogus", "question": "x?", "gold": {},
    }]})

    with pytest.raises(QuestionsFileError, match="invalid source_type"):
        load_questions(path)


def test_load_questions_raises_on_duplicate_question_id(tmp_path):
    path = _write(tmp_path, {"questions": [
        {"question_id": "q1", "source_type": "recall", "question": "a?", "gold": {}},
        {"question_id": "q1", "source_type": "maude", "question": "b?", "gold": {}},
    ]})

    with pytest.raises(QuestionsFileError, match="duplicate question_id"):
        load_questions(path)
