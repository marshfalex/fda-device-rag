def next_page(db, after_created_at, limit: int = 50):
    """Return the next page of posts, newest first, after the given cursor."""
    return db.execute(
        "SELECT id, created_at, body FROM posts "
        "WHERE created_at < ? ORDER BY created_at DESC LIMIT ?",
        (after_created_at, limit),
    ).fetchall()
