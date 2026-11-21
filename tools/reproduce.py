"""Regenerate every result the paper reports and compare it with the committed logs.

    uv run python tools/reproduce.py --check            # rerun, compare with committed logs, change nothing
    uv run python tools/reproduce.py --write            # rerun and overwrite the committed logs
    uv run python tools/reproduce.py --check --only validate

Steps run from the saved fits (intentions_game/results/*.npz), so this takes minutes, not hours (see README for
refitting from scratch). The comparison is exact for text and tolerant for numbers: two numbers agree when they
differ by at most one unit in the last printed digit, or when both are below 1e-8 in absolute value (numerical
validation residuals such as 2.4e-15, which vary across platforms). Timings are ignored.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IG = ROOT / "intentions_game"
NOISE = re.compile(r"^(Unable|.*backend|.*TPU|.*cuda|.*Platform)", re.I)

# (group, working directory, script and arguments, log file)
STEPS = [
    ("validate", IG, ["validate_reference.py"], IG / "results/validate_reference.log"),
    ("validate", IG, ["validate_memo.py"], IG / "results/validate_memo.log"),
    ("validate", IG, ["validate_hier.py"], IG / "results/validate_hier.log"),
    ("validate", IG, ["validate_intent.py"], IG / "results/validate_intent.log"),
    ("validate", IG, ["validate_hier_intent.py"], IG / "results/validate_hier_intent.log"),
    ("numbers", IG, ["bms.py"], IG / "results/bms.log"),
    ("numbers", IG, ["bms_h.py"], IG / "results/bms_h.log"),
    ("numbers", IG, ["pxp_report.py"], IG / "results/pxp_report.log"),
    ("numbers", IG, ["family_report.py"], IG / "results/family_report.log"),
    ("numbers", IG, ["intent_analysis.py"], IG / "results/intent_analysis.log"),
    ("numbers", IG, ["representation_accuracy.py"], IG / "results/representation_accuracy.log"),
    ("numbers", IG, ["split_half.py", "analyze", "3"], IG / "results/split_half_analysis.log"),
    ("numbers", IG, ["recovery_h.py", "analyze"], IG / "results/recovery_H_analysis.log"),
    ("numbers", IG, ["recovery_i.py", "analyze", "2"], IG / "results/recovery_I_analysis.log"),
    ("numbers", IG, ["model_recovery.py", "analyze"], IG / "results/model_recovery_analysis.log"),
    ("numbers", IG, ["coupling_demo.py"], IG / "results/coupling_demo.log"),
    ("numbers", ROOT, ["development/corriveau_trust.py"], ROOT / "development/corriveau_trust.log"),
    ("numbers", ROOT, ["development/corriveau_baselines.py"], ROOT / "development/corriveau_baselines.log"),
    ("numbers", ROOT, ["development/corriveau_routes.py"], ROOT / "development/corriveau_routes.log"),
    ("numbers", ROOT, ["advice_taking/bart_et_advisor.py"], ROOT / "advice_taking/bart_et_advisor.log"),
]
# credulity re-tests: one log, several runs with headers (Section 4.7)
ETMCQ = [("### group winners (default)", []),
         ("### common model M1, HBI params", ["M1", "hbi"]),
         ("### common model M3, HBI params", ["M3", "hbi"]),
         ("### common model M4, HBI params", ["M4", "hbi"]),
         ("### common model M3, our fits", ["M3", "ours"]),
         ("### common model H, our fits", ["H", "ours"])]
FIGURE_EXPORTS = ["figures/scripts/export_fig3_coupling.py", "figures/scripts/export_fig4_intent.py",
                  "figures/scripts/export_fig5_form.py", "figures/scripts/export_fig6_bart.py"]

TIMING = [(re.compile(r"\(\d+(\.\d+)?s incl\. reference\)"), "(T incl. reference)"),
          (re.compile(r"\d+(\.\d+)? ms per call"), "T ms per call"),
          (re.compile(r"\(\d+(\.\d+)?s\)"), "(Ts)")]
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def run(cwd, args):
    out = subprocess.run([sys.executable, "-I", *args], cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{out.stderr[-2000:]}")
    text = out.stdout + out.stderr
    return "\n".join(l for l in text.splitlines() if not NOISE.match(l)).rstrip() + "\n"


def normalise(line):
    for pat, rep in TIMING:
        line = pat.sub(rep, line)
    return line


def decimals(tok):
    mant = re.split(r"[eE]", tok)[0]
    return len(mant.split(".")[1]) if "." in mant else 0


def numbers_agree(a, b):
    x, y = float(a), float(b)
    if abs(x) < 1e-8 and abs(y) < 1e-8:
        return True
    if "e" in a.lower() or "e" in b.lower():
        return abs(x - y) <= 1e-6 * max(abs(x), abs(y))
    return abs(x - y) <= 1.01 * 10 ** (-max(decimals(a), decimals(b)))


def compare(old, new):
    """Return a list of human-readable differences (empty if the texts agree)."""
    a, b = [normalise(l) for l in old.splitlines()], [normalise(l) for l in new.splitlines()]
    if len(a) != len(b):
        return [f"line count {len(a)} -> {len(b)}"]
    diffs = []
    for i, (la, lb) in enumerate(zip(a, b), 1):
        if NUM.sub("#", la) != NUM.sub("#", lb):
            diffs.append(f"line {i}: text differs\n    - {la}\n    + {lb}")
            continue
        for ta, tb in zip(NUM.findall(la), NUM.findall(lb)):
            if not numbers_agree(ta, tb):
                diffs.append(f"line {i}: {ta} -> {tb}\n    - {la}\n    + {lb}")
                break
    return diffs


def compare_csv_dirs(old_dir, new_dir):
    diffs = []
    for f in sorted(Path(old_dir).glob("*")):
        g = Path(new_dir) / f.name
        if not g.exists():
            diffs.append(f"{f.name}: missing after rerun")
            continue
        for d in compare(f.read_text().replace(",", " , "), g.read_text().replace(",", " , ")):
            diffs.append(f"{f.name}: {d}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    ap.add_argument("--only", choices=["validate", "numbers", "figures-data"])
    opt = ap.parse_args()
    failures = 0

    def handle(label, new_text, log):
        nonlocal failures
        if opt.write or not log.exists():
            log.write_text(new_text)
            print(f"  wrote   {log.relative_to(ROOT)}" + ("" if opt.write else "  (new)"))
            return
        diffs = compare(log.read_text(), new_text)
        if diffs:
            failures += 1
            print(f"  DIFFERS {log.relative_to(ROOT)}")
            for d in diffs[:5]:
                print("    " + d.replace("\n", "\n    "))
        else:
            print(f"  ok      {log.relative_to(ROOT)}")

    for group, cwd, args, log in STEPS:
        if opt.only and opt.only != group:
            continue
        handle(group, run(cwd, args), log)
    if not opt.only or opt.only == "numbers":
        text = "".join(f"{head}\n{run(IG, ['etmcq_check.py', *args])}" for head, args in ETMCQ)
        handle("numbers", text, IG / "results/etmcq_check.log")
    if not opt.only or opt.only == "figures-data":
        data = ROOT / "figures/data"
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "data"
            shutil.copytree(data, saved)
            for script in FIGURE_EXPORTS:
                run(ROOT, [script])
            if opt.write:
                print("  wrote   figures/data/*")
            else:
                diffs = compare_csv_dirs(saved, data)
                shutil.rmtree(data)
                shutil.copytree(saved, data)            # --check never changes committed files
                if diffs:
                    failures += 1
                    print("  DIFFERS figures/data")
                    for d in diffs[:5]:
                        print("    " + d.replace("\n", "\n    "))
                else:
                    print("  ok      figures/data/*")
    if opt.check:
        print("\nALL RESULTS REPRODUCED" if failures == 0 else f"\n{failures} RESULT FILE(S) DIFFER")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
