# ─── Reusable greeting function ─────────────────────────────────
# Based on the time-based greeting from My_profile/script.js


def get_greeting(hour):
    """Return a greeting based on the hour (0-23)."""

    # Decide which greeting to use based on the hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


# ─── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(get_greeting(9))    # → Good morning
    print(get_greeting(14))   # → Good afternoon
    print(get_greeting(20))   # → Good evening
