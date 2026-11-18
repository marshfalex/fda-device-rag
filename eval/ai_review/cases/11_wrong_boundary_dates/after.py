from datetime import date


def is_in_trial(signup: date, today: date, trial_days: int = 14) -> bool:
    """True for days 0..trial_days-1 after signup (a 14-day trial)."""
    return (today - signup).days <= trial_days
