import json
from dataclasses import dataclass
from pathlib import Path

VALID_SOURCE_TYPES = {"recall", "maude", "guidance", "ifu"}
REQUIRED_FIELDS = ("question_id", "source_type", "question", "gold")


@dataclass
class Question:
    question_id: str
    source_type: str
    category: str | None
    question: str
    gold: dict
    notes: str | None


class QuestionsFileError(Exception):
    """Raised when the frozen questions file is missing required fields or
    otherwise doesn't match the schema in
    docs/superpowers/specs/2026-08-03-eval-harness-design.md section 7."""


def load_questions(path) -> list[Question]:
    path = Path(path)
    data = json.loads(path.read_text())
    raw_questions = data.get("questions")
    if raw_questions is None:
        raise QuestionsFileError(f"{path}: missing top-level 'questions' key")

    questions = []
    seen_ids = set()
    for i, raw in enumerate(raw_questions):
        for field in REQUIRED_FIELDS:
            if field not in raw:
                raise QuestionsFileError(f"{path}: question at index {i} missing required field {field!r}")

        if raw["source_type"] not in VALID_SOURCE_TYPES:
            raise QuestionsFileError(
                f"{path}: question {raw['question_id']!r} has invalid source_type "
                f"{raw['source_type']!r} (must be one of {sorted(VALID_SOURCE_TYPES)})"
            )

        if raw["question_id"] in seen_ids:
            raise QuestionsFileError(f"{path}: duplicate question_id {raw['question_id']!r}")
        seen_ids.add(raw["question_id"])

        questions.append(Question(
            question_id=raw["question_id"],
            source_type=raw["source_type"],
            category=raw.get("category"),
            question=raw["question"],
            gold=raw["gold"],
            notes=raw.get("notes"),
        ))
    return questions
