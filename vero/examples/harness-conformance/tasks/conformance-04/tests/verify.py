"""Exact-match verifier: strips whitespace, compares as a number."""

from pathlib import Path

EXPECTED = "144"
ANSWER = Path("/app/answer.txt")
REWARD = Path("/logs/verifier/reward.txt")


def main() -> None:
    REWARD.parent.mkdir(parents=True, exist_ok=True)
    text = ANSWER.read_text(encoding="utf-8").strip() if ANSWER.is_file() else ""
    try:
        matched = abs(float(text) - float(EXPECTED)) < 1e-9
    except ValueError:
        matched = False
    REWARD.write_text("1\n" if matched else "0\n", encoding="utf-8")
    print(f"expected={EXPECTED} got={text!r} reward={int(matched)}")


if __name__ == "__main__":
    main()
