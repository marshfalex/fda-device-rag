import subprocess

import pytest

from fda_device_rag.eval.freeze_check import FrozenFileError, assert_frozen_and_get_hash


def _init_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True)


def test_returns_blob_hash_for_a_clean_committed_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')
    subprocess.run(["git", "add", "questions.json"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "freeze"], cwd=tmp_path, capture_output=True, check=True)

    result = assert_frozen_and_get_hash("questions.json", cwd=tmp_path)

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD:questions.json"], cwd=tmp_path, capture_output=True, check=True,
    ).stdout.decode().strip()
    assert result == expected
    assert len(result) == 40


def test_raises_on_uncommitted_modification(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')
    subprocess.run(["git", "add", "questions.json"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "freeze"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "questions.json").write_text('{"questions": [{"edited": true}]}')

    with pytest.raises(FrozenFileError, match="uncommitted changes"):
        assert_frozen_and_get_hash("questions.json", cwd=tmp_path)


def test_raises_when_file_was_never_committed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')

    with pytest.raises(FrozenFileError, match="not committed"):
        assert_frozen_and_get_hash("questions.json", cwd=tmp_path)
