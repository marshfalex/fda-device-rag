import subprocess
from pathlib import Path


class FrozenFileError(Exception):
    """Raised when the frozen questions file isn't actually frozen: either
    the working tree has uncommitted changes to it, or it isn't tracked in
    HEAD at all. This makes 'frozen' a mechanically enforced guarantee
    (design doc section 7), not just a documented promise -- leakage can't
    happen by accident even if the procedural rule is forgotten."""


def assert_frozen_and_get_hash(path, cwd=None) -> str:
    """Raises FrozenFileError if `path` has uncommitted changes relative to
    HEAD (staged or not), or if it isn't committed in HEAD at all. Otherwise
    returns the git blob hash of the committed content, for permanent
    per-eval-run logging (a clean working tree alone can't distinguish a
    genuine one-time freeze from edit-rerun-recommit-rerun)."""
    path = Path(path)

    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)],
        cwd=cwd, capture_output=True,
    )
    if diff.returncode not in (0, 1):
        stderr = diff.stderr.decode().strip()
        # If HEAD doesn't exist (bad revision), the file can't be committed
        if "bad revision" in stderr:
            raise FrozenFileError(
                f"{path} is not committed in HEAD -- commit it before running eval "
                f"({stderr})"
            )
        raise FrozenFileError(
            f"git diff failed checking {path} (exit code {diff.returncode}): {stderr}"
        )
    if diff.returncode == 1:
        raise FrozenFileError(
            f"{path} has uncommitted changes -- commit it before running eval "
            f"(this file must be frozen before any retrieval scoring)"
        )

    hash_result = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.as_posix()}"],
        cwd=cwd, capture_output=True,
    )
    if hash_result.returncode != 0:
        raise FrozenFileError(
            f"{path} is not committed in HEAD -- commit it before running eval "
            f"({hash_result.stderr.decode().strip()})"
        )
    return hash_result.stdout.decode().strip()
