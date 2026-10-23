"""Run the unit tests with the project's Python (used by the GPU worker)."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.exit(subprocess.call([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=root))
