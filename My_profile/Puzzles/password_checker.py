# ─── Password Checker ───────────────────────────────────────────
# Validates a password as Strong, Weak, or Invalid


def check_password(password):
    """Check password strength and return 'Strong', 'Weak', or 'Invalid'."""

    # Too short — not acceptable at all
    if len(password) < 6:
        return "Invalid"

    # Check if it has at least one letter and at least one number
    has_letter = any(char.isalpha() for char in password)
    has_number = any(char.isdigit() for char in password)

    # Both letters and numbers → strong
    if has_letter and has_number:
        return "Strong"

    # Only one type → weak
    return "Weak"


# ─── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(check_password("abc"))       # → Invalid
    print(check_password("abcdef"))    # → Weak
    print(check_password("123456"))    # → Weak
    print(check_password("abc123"))    # → Strong
    print(check_password("hello9"))    # → Strong
