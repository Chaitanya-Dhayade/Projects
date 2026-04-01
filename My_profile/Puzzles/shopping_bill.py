# ─── Shopping Bill Calculator ───────────────────────────────────
# Calculates total with discount and prints a receipt


def calculate_bill(prices):
    """Calculate the total bill with discount based on amount."""

    # Add up all item prices
    total = sum(prices)

    # Decide discount based on total
    if total >= 1000:
        discount = 20
    elif total >= 500:
        discount = 10
    else:
        discount = 0

    # Calculate final amount
    you_pay = total - (total * discount / 100)

    # Print a receipt
    print("--- Receipt ---")
    print(f"Items total : ₹{total:.2f}")
    print(f"Discount    : {discount}%")
    print(f"You pay     : ₹{you_pay:.2f}")

    return you_pay


# ─── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(calculate_bill([100, 150, 200]))   # → 450.0  (no discount)
    print()
    print(calculate_bill([200, 300, 150]))   # → 585.0  (10% off)
    print()
    print(calculate_bill([500, 600, 200]))   # → 1040.0 (20% off)
