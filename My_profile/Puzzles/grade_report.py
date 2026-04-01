# ─── Grade Report ───────────────────────────────────────────────
# Takes a list of student dictionaries and prints a full report


def get_grade(score):
    """Return a letter grade based on the score."""

    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 45:
        return "D"
    else:
        return "F"


def grade_report(students):
    """Print a formatted grade report for a list of students."""

    # Guard against empty list
    if not students:
        print("No students to report.")
        return

    # Calculate class average
    total = sum(s["score"] for s in students)
    average = total / len(students)

    # Find top scorer
    top = max(students, key=lambda s: s["score"])

    # Count failures
    failed = sum(1 for s in students if s["score"] < 45)

    # Print the report
    print("===== Grade Report =====")
    for s in students:
        grade = get_grade(s["score"])
        print(f"{s['name']:<8} → {s['score']:<4} {grade}")

    print("------------------------")
    print(f"Class average  : {average:.1f}")
    print(f"Top scorer     : {top['name']} ({top['score']})")
    print(f"Students failed: {failed}")


# ─── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    grade_report([
        {"name": "Rahul",  "score": 78},
        {"name": "Priya",  "score": 92},
        {"name": "Arjun",  "score": 45},
        {"name": "Sneha",  "score": 61},
        {"name": "Vikram", "score": 38}
    ])
