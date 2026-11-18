def add_tag(tag: str, tags: list = []) -> list:
    """Return a new list of tags including `tag`."""
    tags.append(tag)
    return tags
