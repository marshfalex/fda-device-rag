import threading

stock = {"widget": 10}
_lock = threading.Lock()


def reserve(item: str, qty: int) -> bool:
    """Called concurrently from many request threads."""
    with _lock:
        available = stock.get(item, 0)
        if available >= qty:
            stock[item] = available - qty
            return True
    return False
