"""Several replicas at once on one GPU, one spawned process per replica."""
import multiprocessing as mp
import os, shutil, signal, subprocess, sys, time
from pathlib import Path

from .replicates import run_replicas

try:
    from IPython.display import clear_output
except ImportError:                                  # outside a notebook: plain printing
    def clear_output(wait=False):
        pass


def _worker(log, kwargs):
    """Runs in the child: its own process group, so a notebook interrupt reaches it only
    through run_parallel (one clean signal), and all output, C++ included, in its log."""
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    run_replicas(**kwargs)


def start_mps():
    """Start NVIDIA MPS so the processes' kernels run side by side; True if it is on."""
    if shutil.which("nvidia-cuda-mps-control") is None:
        print("MPS isn't available here: the processes will time-share the GPU")
        return False
    r = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip()
    if r.returncode == 0 or "already" in msg.lower():
        print("MPS is on")
        return True
    print(f"MPS didn't start ({msg or r.returncode}): the processes will time-share the GPU")
    return False


def stop_mps():
    """Shut down the MPS daemon; processes started afterwards use the GPU normally."""
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True)


def _speed(rep_dir, save_name):
    """Latest ns/day in the replica's StateDataReporter CSV (production, else equilibration)."""
    csvs = [p for p in (rep_dir / f"{save_name}.csv", rep_dir / "equil.csv") if p.exists()]
    if not csvs:
        return None
    p = max(csvs, key=lambda q: q.stat().st_mtime)
    with open(p, "rb") as fh:
        fh.seek(max(0, p.stat().st_size - 400))
        lines = fh.read().decode(errors="ignore").strip().splitlines()
    try:
        return float(lines[-1].split(",")[-1])
    except (IndexError, ValueError):
        return None


def _gpu():
    """GPU utilisation (%) and memory used (MiB) from nvidia-smi, as one string."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return r.stdout.strip() or "unavailable"
    except FileNotFoundError:
        return "no nvidia-smi"


def run_parallel(build_dir, reps, mps=True, poll_min=10, stagger_s=20, **kw):
    """Run several replicas at the same time on this machine's GPU, one process each.

    Replica k runs run_replicas(build_dir, only=[k], **kw) in a freshly spawned process,
    which imports the package anew (module edits apply at the next launch) and has its own
    SESSION id. The call blocks, printing a status block every `poll_min` minutes, until
    every process has ended.

    Parameters
    ----------
    build_dir : path-like
        Build folder, as for run_replicas.
    reps : iterable of int
        Replica numbers to run now, one process each (duplicates ignored).
    mps : bool
        Try to start NVIDIA MPS so the processes' GPU kernels run side by side.
    poll_min : float
        Minutes between status blocks.
    stagger_s : float
        Seconds between process starts.
    **kw
        Passed to run_replicas (ns, max_hours, reserve_min, save_name, local_root,
        remote_root, equil, ...); local_root and save_name also locate each replica's CSV
        for the speed readout.

    Files written
    -------------
    - <local_root>/logs/rXX.log:   everything the replica prints, C++ output included

    Reading the status block
    ------------------------
    GPU utilisation and memory, then each replica's state, current ns/day and last log
    line, then their sum. Compare the sum with one replica's speed alone: another process
    is worth adding while the sum keeps rising.

    Notes
    -----
    Interrupting the call sends each replica one SIGINT: production checkpoints, mirrors
    and exits as at its deadline; an equilibration in progress is abandoned and redone at
    the next launch. The processes run in their own process groups and survive a kernel
    restart; stop leftovers with `pkill -INT -f spawn_main`.
    """
    build_dir, reps = Path(build_dir), sorted(set(reps))
    local_root = Path(kw.setdefault("local_root", "/content/replicas"))
    save_name = kw.setdefault("save_name", "md")
    log_dir = local_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENMM_CPU_THREADS": str(max(1, (os.cpu_count() or 2) // len(reps)))}
    if mps and start_mps():
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(min(100, 200 // len(reps)))
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)                   # each child copies the environment as it starts

    ctx, procs = mp.get_context("spawn"), {}
    try:
        for k in reps:
            name = f"r{k:02d}"
            log = log_dir / f"{name}.log"
            p = ctx.Process(target=_worker, name=name,
                            args=(str(log), dict(kw, build_dir=build_dir, only=[k])))
            p.start()
            procs[name] = p
            print(f"{name}: pid {p.pid}, log {log}")
            time.sleep(stagger_s)            # stagger the Drive reads and CUDA start-up
    finally:
        for k, v in saved.items():           # leave the notebook's environment as it was
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    t0, shown = time.time(), 0.0
    try:
        while any(p.is_alive() for p in procs.values()):
            time.sleep(30)
            if time.time() - shown < poll_min * 60:
                continue
            shown = time.time()
            clear_output(wait=True)
            print(f"{(shown - t0) / 3600:.1f} h in, GPU utilisation %, memory MiB: {_gpu()}")
            total = 0.0
            for name, p in procs.items():
                v = _speed(local_root / build_dir.name / name, save_name) or 0.0
                total += v if p.is_alive() else 0.0
                tail = (log_dir / f"{name}.log").read_text(errors="ignore").strip().splitlines()[-1:]
                state = "running" if p.is_alive() else f"exit {p.exitcode}"
                print(f"  {name}  {state:<8} {v:7.1f} ns/day   {tail[0][:80] if tail else ''}")
            print(f"  together: {total:.1f} ns/day")
    except KeyboardInterrupt:
        print("stopping: each replica checkpoints and mirrors before it exits")
        for p in procs.values():
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)    # produce() treats this like its deadline
        for name, p in procs.items():
            p.join(timeout=900)
            if p.is_alive():
                print(f"  {name} is still finishing; check its log")
    codes = {name: p.exitcode for name, p in procs.items()}
    print("exit codes:", codes)
    return codes