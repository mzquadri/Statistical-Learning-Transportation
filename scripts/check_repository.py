"""Fail if the README stops agreeing with verification/verification.json.

    python scripts/check_repository.py

Checks that the documented files exist, that the verification still passed, that
the structural claims the README makes are true of the repository, and that
every count the README quotes still matches the recorded artifact.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "verification" / "verification.json"

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "docs/diagrams/coursework.svg",
    "scripts/verify_results.py",
    "scripts/check_repository.py",
    "tests/test_artifacts.py",
    "verification/verification.json",
    "Assignment_1/README.md",
    "Assignment_1/requirements.txt",
    "Assignment_1/Problem_Set_1_Notebook.ipynb",
    "Assignment_1/data_problem_3.csv",
    "Assignment_1/data_problem_4.csv",
    "Assignment_2/README.md",
    "Assignment_2/requirements.txt",
    "Assignment_2/Problem_Set_2_Notebook.ipynb",
    "Assignment_2/Problem_Set_2_Report.pdf",
    "Assignment_2/Problem_Set_2_Report.tex",
    "Assignment_2/data/mode_choice_pcasample.csv",
    "Assignment_3/mlp.py",
    "Assignment_3/solution/README.md",
    "Assignment_3/solution/requirements.txt",
    "Assignment_3/solution/mlp.py",
    "Assignment_3/solution/Problem_Set_3_Notebook.ipynb",
    "Assignment_3/solution/Problem_Set_3_Report.pdf",
    "Assignment_3/solution/Problem_Set_3_Report.tex",
    "Assignment_3/solution/results/reported_values.csv",
    "Assignment_3/solution/data/nyc-yellow-may2oct.csv",
)


def structural_checks() -> list[str]:
    """Claims about the repository, checked against it rather than asserted."""
    notes = []

    # The README says Problem Set 1's report is withheld. If it ever appears,
    # the matriculation number comes with it.
    withheld = ROOT / "Assignment_1" / "Problem_Set_1_Report.pdf"
    if withheld.exists():
        raise SystemExit("  Assignment_1/Problem_Set_1_Report.pdf is present. It "
                         "carries the matriculation number and the README says "
                         "it is withheld.")
    notes.append("  the Problem Set 1 report is still withheld")

    # The supplied skeleton and the completed file must stay distinguishable.
    supplied = (ROOT / "Assignment_3" / "mlp.py").read_text(encoding="utf-8")
    completed = (ROOT / "Assignment_3" / "solution" / "mlp.py").read_text(
        encoding="utf-8")
    if supplied == completed:
        raise SystemExit("  the supplied mlp.py skeleton and the completed file "
                         "are identical; starter code must stay separable")
    gaps = supplied.count("### YOUR CODE STARTS HERE ###")
    if gaps != completed.count("### YOUR CODE STARTS HERE ###"):
        raise SystemExit("  the completed mlp.py no longer marks which regions "
                         "were the skeleton's gaps")
    notes.append(f"  the supplied skeleton's {gaps} gaps are still marked in the "
                 f"completed file")

    # One copy of the taxi data, not two.
    copies = sorted(p.relative_to(ROOT).as_posix()
                    for p in ROOT.rglob("nyc-yellow-may2oct.csv"))
    if len(copies) != 1:
        raise SystemExit(f"  the taxi dataset is committed {len(copies)} times: "
                         f"{', '.join(copies)}")
    notes.append("  the taxi dataset is committed once")

    # Every assignment folder carries what the README says it carries.
    for folder in ("Assignment_1", "Assignment_2", "Assignment_3/solution"):
        for name in ("README.md", "requirements.txt"):
            if not (ROOT / folder / name).is_file():
                raise SystemExit(f"  {folder}/{name} is missing, but the README "
                                 f"says every assignment folder carries one")
    notes.append("  all three assignment folders carry a README and requirements")

    # No matriculation number anywhere in the text files.
    pattern = re.compile(r"\b0\d{7}\b")
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix not in {".md", ".py", ".tex", ".txt", ".yml"}:
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8",
                                                     errors="replace")):
            raise SystemExit(f"  {path.relative_to(ROOT)} contains what looks "
                             f"like a matriculation number: {match.group(0)}")
    notes.append("  no matriculation number in any tracked text file")
    return notes


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    totals = data["totals"]
    by_set = data["by_assignment"]
    binding = data["report_binding"]

    out = [
        (f"recomputes {totals['checks']} claims", "the total check count"),
        (f"| Problem Set 1 | {by_set['Problem Set 1']['checks']} | "
         f"{by_set['Problem Set 1']['passed']} |", "the Problem Set 1 row"),
        (f"| Problem Set 2 | {by_set['Problem Set 2']['checks']} | "
         f"{by_set['Problem Set 2']['passed']} |", "the Problem Set 2 row"),
        (f"| Problem Set 3 | {by_set['Problem Set 3']['checks']} | "
         f"{by_set['Problem Set 3']['passed']} |", "the Problem Set 3 row"),
    ]
    if binding.get("available"):
        out.append((f"All {binding['registry_values']} Problem Set 3 registry "
                    f"values appear verbatim", "the registry binding"))
    return out


def main() -> int:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")
    print(f"  {len(REQUIRED_FILES)} required files present")

    if not VERIFICATION.is_file():
        raise SystemExit("  verification/verification.json is missing; run "
                         "python scripts/verify_results.py")
    data = json.loads(VERIFICATION.read_text(encoding="utf-8"))

    if data["failed"]:
        raise SystemExit(f"  the recorded verification has "
                         f"{len(data['failed'])} failing checks")
    totals = data["totals"]
    if totals["passed"] != totals["checks"]:
        raise SystemExit("  the recorded verification did not pass every check")
    print(f"  the recorded verification passed {totals['passed']} of "
          f"{totals['checks']} checks")

    for note in structural_checks():
        print(note)

    def flatten(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    flat = flatten((ROOT / "README.md").read_text(encoding="utf-8"))
    checks = claims(data)
    failures = [f"  README does not state {description}: expected "
                f"{flatten(expected)!r}"
                for expected, description in checks
                if flatten(expected) not in flat]
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    if failures:
        print()
        for failure in failures:
            print(failure)
        plural = "claim" if len(failures) == 1 else "claims"
        raise SystemExit(f"\n  {len(failures)} {plural} in the README no longer "
                         f"match verification/verification.json")

    print("  README and verification/verification.json agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
