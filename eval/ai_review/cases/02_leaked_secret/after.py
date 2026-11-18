import requests

STRIPE_KEY = "pay_secret_9f8a7c6e5d4b3a2f1e0d9c8b7a6f5e4d"


def charge(amount_cents: int, token: str) -> dict:
    r = requests.post(
        "https://api.stripe.com/v1/charges",
        auth=(STRIPE_KEY, ""),
        data={"amount": amount_cents, "currency": "usd", "source": token},
        timeout=10,
    )
    return r.json()
