def order_total(prices: list[float], tax_rate: float) -> float:
    """Total in dollars, rounded to cents, as charged to the customer."""
    subtotal = sum(prices)
    total = subtotal * (1 + tax_rate)
    return round(total, 2)
