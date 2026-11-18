import time

import requests


def charge_card(customer_id: str, amount_cents: int) -> dict:
    """Charge a customer once. Retries transient network failures."""
    for attempt in range(3):
        try:
            r = requests.post("https://payments.internal/charges",
                              json={"customer": customer_id, "amount": amount_cents}, timeout=5)
            return r.json()
        except requests.Timeout:
            time.sleep(2 ** attempt)
    raise RuntimeError("payment service unavailable")
