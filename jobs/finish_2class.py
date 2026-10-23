"""Finish the 2-class (rest vs task) results once training is complete.

Moves outputs/v3_2class to results/v3_2class, rebuilds both summaries, verifies both folders and
builds the before/after table and chart. Safe to re-run.

    .\\.venv\\Scripts\\python.exe -B jobs\\finish_2class.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print(">", " ".join(args), flush=True)
    code = subprocess.call([sys.executable, "-B", *args], cwd=ROOT)
    if code:
        raise SystemExit(f"step failed ({code}): {' '.join(args)}")


def main():
    src, dst = ROOT / "outputs" / "v3_2class", ROOT / "results" / "v3_2class"
    after = (dst if dst.exists() else src) / "after"
    n = len(list((after / "parts").glob("*.json")))
    if n < 60:
        raise SystemExit(f"2-class training not finished yet: {n}/60 results in {after}")
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
    for d in ["after", "before_v1like"]:
        claims = dst / d / "claims"
        if claims.exists():
            target = ROOT / "DELETE" / "outputs" / "claims_2class" / d
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(claims), str(target))
    run("benchmark_v3.py", "--task", "2class", "--output", "results/v3_2class/after", "--variants", "filtered-ea-sag",
        "--summarize", "--device", "cpu")
    run("benchmark_v3.py", "--task", "2class", "--output", "results/v3_2class/before_v1like", "--variants", "raw",
        "--summarize", "--device", "cpu")
    run("verify_v3.py", "results/v3_2class/after")
    run("verify_v3.py", "results/v3_2class/before_v1like")
    run("make_slide_table.py", "--before", "results/v3_2class/before_v1like/summary.csv", "--before-variant", "raw",
        "--after", "results/v3_2class/after/summary.csv", "--after-variant", "filtered-ea-sag",
        "--out", "results/v3_2class", "--title", "STEW 2-class rest vs task, unseen-subject 5-fold CV")
    print("2-class results ready in results/v3_2class (BEFORE_AFTER.md, before_after.png)", flush=True)


if __name__ == "__main__":
    main()
