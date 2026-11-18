def paginate(items: list, page: int, per_page: int) -> list:
    """Return the 1-indexed `page` of `items`."""
    if page < 1:
        raise ValueError("page must be >= 1")
    start = page * per_page
    return items[start:start + per_page]
