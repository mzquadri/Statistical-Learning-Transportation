"""Builds the Moodle submission archive. Development infrastructure - NOT in the ZIP.

The archive name follows the problem sheet: "Matriculation number, Name.zip".
Contents are listed explicitly rather than globbed from the folder, so a stray
file in the working directory can never be swept into the submission.
"""
import os
import zipfile
from pathlib import Path

# Matriculation number withheld from the public repository.
# Set MATRIC in the environment when rebuilding the submission archive.
MATRIC, NAME = os.environ.get("MATRIC", "MATRIC"), "Mohd Zamin Quadri"
SRC = Path(__file__).resolve().parent
OUT = SRC.parent / f"{MATRIC}, {NAME}.zip"

FILES = [
    "Problem_Set_3_Report.pdf",       # the report, covering Problems 1-4
    "Problem_Set_3_Notebook.ipynb",   # all code, executed, outputs stored
    "mlp.py",                         # supplied skeleton, completed (Problem 1)
    "README.md",
    "requirements.txt",
    "data/nyc-yellow-may2oct.csv",    # data supplied with the assignment
]
# Every table the notebook writes, so each number in the report can be traced.
# Accidental file-manager duplicates ("... - Copy.csv", "...(1).csv") are excluded:
# one appeared in results/ during review and would otherwise have been swept in.
JUNK = (" - copy", "copy of ", "(1)", "(2)", "-copy", "_copy", ".bak", "~")
_csv = [p for p in (SRC / "results").glob("*.csv")
        if not any(j in p.name.lower() for j in JUNK)]
_skipped = [p.name for p in (SRC / "results").glob("*.csv")
            if any(j in p.name.lower() for j in JUNK)]
if _skipped:
    print(f"skipped {len(_skipped)} duplicate/backup file(s) in results/: {_skipped}")
FILES += sorted(f"results/{p.name}" for p in _csv)

EXPECTED = 29
if len(FILES) != EXPECTED:
    raise SystemExit(f"expected {EXPECTED} entries, assembled {len(FILES)}:\n  "
                     + "\n  ".join(FILES))

missing = [f for f in FILES if not (SRC / f).exists()]
if missing:
    raise SystemExit(f"refusing to build, missing: {missing}")

if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(SRC / f, arcname=f)

with zipfile.ZipFile(OUT) as z:
    bad = z.testzip()
    assert bad is None, f"corrupt entry: {bad}"
    names = z.namelist()
assert names == FILES, "archive contents differ from the declared list"
print(f"wrote {OUT.name}  ({OUT.stat().st_size:,} bytes, {len(names)} entries)")
for n in names:
    print(f"  {n}")
