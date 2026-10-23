"""GPU job worker for the STEW project.

Start it once from the project folder and leave the window open:

    .\\.venv\\Scripts\\python.exe -u jobs\\gpu_worker.py

It watches jobs\\queue for job files (*.json), runs each one with this same Python
(so it uses the project's .venv and the GPU), and writes the output to jobs\\logs.
It only runs Python scripts that live inside this project folder.
Stop it by closing the window, pressing Ctrl+C, or creating a file named jobs\\STOP.

Job file format:  {"script": "train_v3.py", "args": ["--output", "results/x"], "lane": "gpu"}
Settings (optional) in jobs\\config.json:  {"max_parallel": 2}
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
QUEUE, RUNNING, DONE, FAILED, LOGS = (JOBS / d for d in ["queue", "running", "done", "failed", "logs"])
for d in [QUEUE, RUNNING, DONE, FAILED, LOGS]:
    d.mkdir(parents=True, exist_ok=True)
STATUS = JOBS / "worker_status.json"


def now():
    return datetime.now().isoformat(timespec="seconds")


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi unavailable: {e}"


def max_parallel():
    try:
        return int(json.loads((JOBS / "config.json").read_text(encoding="utf-8")).get("max_parallel", 2))
    except Exception:  # noqa: BLE001
        return 2


def start(job_file):
    name = job_file.stem
    try:
        job = json.loads(job_file.read_text(encoding="utf-8"))
        script = (ROOT / job["script"]).resolve()
        if ROOT not in script.parents or script.suffix != ".py" or not script.exists():
            raise ValueError(f"refusing to run {script}: must be an existing .py file inside {ROOT}")
        args = [str(a) for a in job.get("args", [])]
    except Exception as e:  # noqa: BLE001
        (LOGS / f"{name}.log").write_text(f"[{now()}] bad job file: {e}\n", encoding="utf-8")
        job_file.replace(FAILED / job_file.name)
        return None
    running_file = RUNNING / job_file.name
    job_file.replace(running_file)
    log = open(LOGS / f"{name}.log", "a", encoding="utf-8", buffering=1)
    log.write(f"[{now()}] start: {script.name} {' '.join(args)}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-u", "-B", str(script), *args], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, env=env, creationflags=flags)
    print(f"[{now()}] started {name} (pid {proc.pid})", flush=True)
    return {"name": name, "proc": proc, "log": log, "file": running_file, "started": now()}


def keep_awake(on=True):
    """Ask Windows not to sleep while the worker runs (no settings are changed; ends with the worker)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception as e:  # noqa: BLE001
        print(f"(could not set keep-awake: {e})", flush=True)


def main():
    keep_awake(True)
    # Jobs left in running/ by a previous worker that was closed go back to the queue.
    for f in RUNNING.glob("*.json"):
        f.replace(QUEUE / f.name)
    print(f"[{now()}] STEW GPU worker ready. Watching {QUEUE}", flush=True)
    print(f"GPU: {gpu_info()}", flush=True)
    print("Leave this window open (minimising is fine). The PC is kept awake while this runs,", flush=True)
    print("but closing the laptop lid may still put it to sleep. Close this window or press Ctrl+C to stop.", flush=True)
    active = []
    last_status = 0
    while not (JOBS / "STOP").exists():
        for a in list(active):
            code = a["proc"].poll()
            if code is not None:
                a["log"].write(f"[{now()}] exit code {code}\n")
                a["log"].close()
                a["file"].replace((DONE if code == 0 else FAILED) / a["file"].name)
                print(f"[{now()}] finished {a['name']} -> {'done' if code == 0 else 'FAILED'} (code {code})", flush=True)
                active.remove(a)
        while len(active) < max_parallel():
            pending = sorted(QUEUE.glob("*.json"))
            if not pending:
                break
            a = start(pending[0])
            if a:
                active.append(a)
        if time.time() - last_status > 15:
            STATUS.write_text(json.dumps({"heartbeat": now(), "python": sys.executable, "gpu": gpu_info(),
                                          "running": [{"name": a["name"], "pid": a["proc"].pid, "started": a["started"]}
                                                      for a in active],
                                          "queued": [f.name for f in sorted(QUEUE.glob("*.json"))],
                                          "max_parallel": max_parallel()}, indent=2), encoding="utf-8")
            last_status = time.time()
        time.sleep(3)
    print(f"[{now()}] STOP file found; waiting for running jobs to finish...", flush=True)
    for a in active:
        a["proc"].wait()
        a["log"].close()
        a["file"].replace(DONE / a["file"].name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
