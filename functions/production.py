from .constants import *
from .colab_funcs import _mirror
from .system_builder import load_model
from .equilibration import _add_barostat, _simulation
import time
import signal
import openmm as mm
import mdtraj as md
import openmm.app as app
import numpy as np
from openmm import unit

def _truncate_csv(path, step):
    """Drop StateDataReporter rows written after the checkpoint being resumed."""
    if not path.exists():
        return

    lines = path.read_text().splitlines()
    if not lines:
        return

    head, body = lines[0], lines[1:]
    kept = [l for l in body if l.strip() and int(l.split(",")[0]) <= step]
    if len(kept) != len(body):
        print(f"  truncating {path.name}: {len(body)} -> {len(kept)} rows")
        path.write_text("\n".join([head] + kept) + "\n")

def _count_frames(path, topology, fmt, subset=None):
    top = md.Topology.from_openmm(topology)
    if subset is not None:
        top = top.subset(subset)
    try:
        return md.load(str(path), top=top).n_frames
    except Exception:
        return None

def _truncate_traj(path, keep, topology, fmt, subset=None):
    """Drop frames written after the checkpoint the run is resuming from."""

    top = md.Topology.from_openmm(topology)
    if subset is not None:
        top = top.subset(subset)
    t = md.load(str(path), top=top)
    if t.n_frames <= keep:
        return
    print(f"  truncating {path.name}: {t.n_frames} -> {keep} frames "
          f"({t.n_frames - keep} written after the last checkpoint)")
    path.unlink()
    if keep:
        t[:keep].save(str(path))

class TrajectoryWatchdog:
    """Fail loudly when the trajectory writer stalls.

    xtc_write_frame opens/appends/closes the file per frame and its return
    value is never checked in Python, so a failing append is SILENT.

    Worse, a stall survives the resume logic. _truncate_traj only trims frames
    ahead of the checkpoint, so a short file passes straight through and the
    reporter appends after it, leaving a gap in the middle that XTC's
    sequential time renumbering makes invisible.

    This checks that the file grew since the last look. O(1), one stat call.
    """

    def __init__(self, path, interval, grace=2):
        self.path, self._interval, self.grace = Path(path), interval, grace
        self._last, self._strikes = -1, 0

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return {"steps": steps, "periodic": None, "include": []}

    def report(self, simulation, state):
        size = self.path.stat().st_size if self.path.exists() else 0
        if size <= self._last:
            self._strikes += 1
            print(f"  !! {self.path.name} has not grown in "
                  f"{self._strikes} check(s) ({size:,} bytes) at step "
                  f"{simulation.currentStep:,}")
            if self._strikes >= self.grace:
                raise RuntimeError(
                    f"trajectory writer has stalled: {self.path} stuck at "
                    f"{size:,} bytes. Almost certainly the output path cannot "
                    f"take appends (Google Drive mount, quota, or full "
                    f"filesystem). Run on local disk and mirror instead. "
                    f"The checkpoint is intact. Fix the path and resume.")
        else:
            self._strikes = 0
        self._last = size


class AtomicCheckpointReporter(app.CheckpointReporter):
    """Checkpoint via temp-file + rename, keeping one generation of backup.

    OpenMM's CheckpointReporter calls saveCheckpoint(filename), which TRUNCATES
    the existing checkpoint before writing the new one. The old good checkpoint
    is destroyed before the new one exists. On local disk that window is
    milliseconds; on a Google Drive mount, it can be seconds and a Colab session
    killed inside that window loses the entire run.

    The new checkpoint is written to .tmp, flushed, then renamed over the
    old one, which is first kept as .bak. A kill at any point leaves either the
    previous checkpoint or the new one intact.
    """

    def __init__(self, file, reportInterval):
        super().__init__(str(file) + ".tmp", reportInterval)
        self._target = Path(file)

    def report(self, simulation, state):
        tmp = Path(str(self._target) + ".tmp")
        super().report(simulation, state)
        try:
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        if self._target.exists():
            bak = Path(str(self._target) + ".bak")
            bak.unlink(missing_ok=True)
            self._target.rename(bak)
        tmp.rename(self._target)


def load_checkpoint_safely(sim, chk):
    """Try the checkpoint, fall back to the backup generation."""
    for path, label in ((chk, "checkpoint"), (Path(str(chk) + ".bak"),
                                              "backup checkpoint")):
        if not path.exists():
            continue
        try:
            sim.loadCheckpoint(str(path))
            # loadCheckpoint does NOT raise on a truncated file. Measured: with a
            # checkpoint cut to 5 kB -> loaded cleanly and then produced
            # "Particle coordinate is NaN" several steps into a previous run.
            # Validate the restored state explicitly rather than trusting the load.
            st = sim.context.getState(getPositions=True, getEnergy=True)
            xyz = np.array(st.getPositions().value_in_unit(unit.nanometer))
            e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not np.isfinite(xyz).all():
                raise ValueError("restored positions contain NaN/inf")
            if not np.isfinite(e):
                raise ValueError("restored potential energy is not finite")
            box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            if np.diag(box).min() <= 0:
                raise ValueError(f"restored box is degenerate: {np.diag(box)}")
            print(f"  resumed from {label} at step {sim.currentStep:,} "
                  f"(E = {e:,.0f} kJ/mol, box OK)")
            return True
        except Exception as e:
            print(f"  !! {label} unusable ({type(e).__name__}: {str(e)[:80]})")
    return False


def produce(out, sys_name="system", model_name="model", equil_name="equil",
            save_name="md", ns=200, temp=310.15, dt_fs=4.0, save_ps=10, seed=1,
            fmt="xtc", keep_water=False, max_hours=None, reserve_min=10.0,
            mirror_to=None, mirror_every_min=60.0):
    """One trajectory, resumable. Rerun the same command after a timeout and it
    continues from the checkpoint: no blocks, no pieces, no filename parsing."""

    TRAJ = {"xtc": (app.XTCReporter, f"{save_name}.xtc"),    # ~0.2 MB/frame at
                                                             # 56k atoms
            "dcd": (app.DCDReporter, f"{save_name}.dcd")}    # ~0.67 MB/frame

    #Loading up the system
    out = Path(out)

    system, model = load_model(out, sys_name, model_name)
    topology = model.topology

    _add_barostat(system, temp)

    sim, _ = _simulation(system, topology, temp, dt_fs, seed)

    chk = out / f"{save_name}.chk"
    resume = chk.exists() or Path(str(chk) + ".bak").exists()

    if resume and load_checkpoint_safely(sim, chk):
        pass
    elif resume:
        sim.loadCheckpoint(str(chk))
        raise RuntimeError(f"{chk} and its backup are both unusable; cannot resume")
    else:
        # NOT loadState(): the equilibration state carries the global parameter
        # `k` from the restraint force, and setState() rejects parameters the
        # system does not define. Transfer positions, velocities and box only.
        st = mm.XmlSerializer.deserialize((out / f"{equil_name}.xml").read_text())
        sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        sim.context.setPositions(st.getPositions())
        try:
            sim.context.setVelocities(st.getVelocities())
        except Exception:
            sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)

    total = int(ns * 1e6 / dt_fs)
    todo = total - sim.currentStep
    if todo <= 0:
        print(f"already complete ({sim.currentStep:,} steps)")
        return sim

    nsav = max(1, int(save_ps * 1000 / dt_fs))
    Reporter, fname = TRAJ[fmt]
    traj_path = out / fname

    # Write only the solute unless water is explicitly wanted.
    subset = None
    if not keep_water:
        subset = [a.index for a in topology.atoms()
                  if a.residue.name.strip() not in SOLVENT_FOR_TRAJ]
        print(f"  trajectory: {len(subset):,} of {topology.getNumAtoms():,} atoms "
              f"(water excluded)")

    # Trajectory and checkpoint are INDEPENDENT reporters, so the trajectory
    # runs ahead of the last checkpoint. On resume, the context rewinds to the
    # checkpoint but the trajectory is appended to, and the overlap is
    # re-simulated and written twice. Timestamps are renumbered sequentially,
    # so the duplication is invisible in the file.

    if resume:
        if traj_path.exists():
          expected = sim.currentStep // nsav
          actual = _count_frames(traj_path, topology, fmt, subset)
          if actual is not None and actual < expected - 1:
                raise RuntimeError(
                    f"{traj_path.name} holds {actual} frames but the checkpoint "
                    f"is at step {sim.currentStep:,} ({expected} frames). The "
                    f"writer stalled during the previous run and appending now "
                    f"would leave an invisible {(expected-actual)*save_ps/1000:.2f} ns "
                    f"gap. Move the stub aside and restart the trajectory, or "
                    f"rewind to a checkpoint that matches.")

          _truncate_traj(traj_path, expected, topology, fmt, subset)

        # the CSV has exactly the same problem as the frame data file
        _truncate_csv(out / f"{save_name}.csv", sim.currentStep)

    # enforcePeriodicBox=False: wrapping splits the duplex across the box
    # boundary and silently corrupts every inter-strand distance downstream.
    sim.reporters += [
        Reporter(str(traj_path), nsav, append=resume, enforcePeriodicBox=False,
                 atomSubset=subset),
        AtomicCheckpointReporter(chk, nsav * 10),
        TrajectoryWatchdog(traj_path, nsav * 10),
        app.StateDataReporter(str(out / f"{save_name}.csv"), nsav,
                              append=resume, step=True,
                              time=True, potentialEnergy=True, temperature=True,
                              density=True, speed=True),
    ]

    print(f"running {todo * dt_fs / 1e6:.1f} ns of {ns} ns")

    # Wall-clock deadline. A Colab session dies without warning, and anything
    # since the last checkpoint is lost. Stopping deliberately with `reserve_min`
    # to spare means the final checkpoint, state and Drive sync all complete.
    deadline = None
    if max_hours:
        deadline = time.time() + max_hours * 3600 - reserve_min * 60
        print(f"  wall-clock limit {max_hours} h "
              f"(stopping {reserve_min:g} min early to save cleanly)")

    stop = {"now": False, "why": ""}

    def _signal(signum, _frame):
        stop["now"] = True
        stop["why"] = f"signal {signum}"


    handlers = {}
    for sig in ("SIGTERM", "SIGINT"):
        try:                                    # only works in the main thread
            n_ = getattr(signal, sig)
            handlers[n_] = signal.signal(n_, _signal)
        except (ValueError, AttributeError, OSError):
            pass

    t0 = time.time()
    last_mirror = time.time()
    try:
        while sim.currentStep < total and not stop["now"]:
            sim.step(min(nsav, total - sim.currentStep))
            if deadline and time.time() > deadline:
                stop["now"], stop["why"] = True, "wall-clock limit"
            if mirror_to and time.time() - last_mirror > mirror_every_min * 60:
                _mirror(out, mirror_to)
                last_mirror = time.time()

    finally:
        for n_, h in handlers.items():
            try:
                signal.signal(n_, h)
            except Exception:
                pass
        # explicit final checkpoint + portable state, whatever the exit route
        AtomicCheckpointReporter(chk, 1).report(sim, sim.context.getState())
        sim.saveState(str(out / f"{save_name}.xml"))

        if mirror_to:                       # always runs: deadline, signal or crash
            _mirror(out, mirror_to)

    done_ns = sim.currentStep * dt_fs / 1e6          # steps * fs -> ns
    el = (time.time() - t0) / 3600
    rate = (done_ns - (total - todo) * dt_fs / 1e6) / max(el, 1e-9) * 24
    if stop["now"]:
      print(f"\n  STOPPED EARLY ({stop['why']}) at {done_ns:.4g} ns of {ns} ns "
            f"({100*done_ns/ns:.1f}%)")
      print(f"  observed {rate:,.0f} ns/day -> {(ns-done_ns)/max(rate,1e-9)*24:.1f} h remaining")
      print(f"  checkpoint written at step {sim.currentStep:,}; rerun the same"
            f" command to continue")
    else:
      print(f"\n  COMPLETE: {done_ns:.4g} ns at {rate:,.0f} ns/day")

    return sim