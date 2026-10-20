"""Independent replicas of one build, each with its own equilibration and production,
resumable across sessions and shared between Colab accounts through Drive."""

from .production import produce
from .system_builder import check_box
from .colab_funcs import _mirror, restore_from_drive
from .equilibration import Equilibrator
import gc, hashlib, time, json, socket, uuid, shutil
from pathlib import Path
import openmm as mm
from scipy.spatial.distance import pdist

SESSION = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"   # identifies this Colab session
print(f"SESSION ID:{SESSION}")

def _claim(remote, hours):
    """Best-effort lock so two sessions don't run the same replica."""
    f = remote / "_lock.json"
    try:
        lock = json.loads(f.read_text())
        if lock["owner"] != SESSION and lock["until"] > time.time():
            return False
    except (OSError, ValueError, KeyError):
        pass
    f.write_text(json.dumps(dict(owner=SESSION, until=time.time() + hours * 3600 + 1800)))
    return True


def _release(remote):
    """Delete the replica's _lock.json if this session owns it."""
    f = remote / "_lock.json"
    try:
        if json.loads(f.read_text())["owner"] == SESSION:
            f.unlink()
    except (OSError, ValueError, KeyError):
        pass


def run_replicas(build_dir, n_rep=3, only=None, ns=200, max_hours=3, reserve_min=15,
                 mirror_every_min=60, save_name="md", seed0=1000, temp=310.15, dt_fs=4.0,
                 save_ps=10, local_root="/content/replicas", remote_root=None,
                 cleanup_local=False, equil=None, system="system.xml", model="model.cif",
                 equil_file="equil.xml"):
    
    """Run independent replicas of one build, one after another, within a time budget.

    Each replica gets a copy of the build (identified by a hash of system.xml), its own
    equilibration and its own production, seeded with seed0 + k (equilibration) and
    seed0 + 500 + k (production). Rerunning the same call resumes unfinished replicas and
    skips finished ones, so the same cell serves every new session.

    Parameters
    ----------
    build_dir : path-like
        Build folder from build_system (on Drive in Colab).
    n_rep : int
        Replicas r01 .. r<n_rep> to run when `only` is not given.
    only : list of int, optional
        Replica numbers for this session; give each Colab account its own list.
    ns, temp, dt_fs, save_ps : float
        Production length (ns), temperature (K), time step (fs), frame interval (ps).
        Recorded in _replica.json at a replica's first start; later calls reuse them.
    max_hours : float
        Wall-clock budget; no replica is started with less than reserve_min + 30 min left.
    reserve_min, mirror_every_min : float
        Passed to produce (clean-stop margin, Drive mirroring interval).
    save_name : str
        Production file stem in each replica folder (the `traj` of analyze_replicas).
    seed0 : int
        Base of the seeds; change it to generate a new, independent set.
    local_root : path-like
        Local working area; replica k lives in <local_root>/<build name>/rXX.
    remote_root : path-like, optional
        Drive folder holding the replicas (default <build_dir>/replicas); use a separate
        one per protocol you want analysed separately.
    cleanup_local : bool
        Delete a replica's local copy once complete.
    equil : Equilibrator, optional
        Equilibration protocol; the default ladder when None.
    system, model, equil_file : str
        File names of the build and of the equilibrated state.

    Files written per replica (<remote_root>/rXX, mirrored from the local copy)
    --------------------------------------------------------------------------
    copies of system.xml, model.cif and build.json; equil.* (equilibrate); <save_name>.*
    (produce); _replica.json (build hash, seeds, settings, ladder, OpenMM version);
    _manifest.json (sync_dir). On Drive only: _lock.json while a session works on the
    replica, _done.json once its production is complete (only those are pooled).
    """
    if equil == None:
        equil = Equilibrator()

    t0 = time.time()
    hours_left = lambda: max_hours - (time.time() - t0) / 3600
    build_dir = Path(build_dir)

    if not (build_dir / system).exists():
        raise FileNotFoundError(f"no {system} in {build_dir}; is folder mounted?")
    remote_root = Path(remote_root or build_dir / "replicas")
    remote_root.mkdir(exist_ok=True)
    local_root = Path(local_root) / build_dir.name
    build_id = hashlib.sha1((build_dir / system).read_bytes()).hexdigest()[:12]
    check_box(build_dir)

    for k in (only or range(1, n_rep + 1)):
        name = f"r{k:02d}"
        local, remote = local_root / name, remote_root / name
        if (remote / "_done.json").exists():
            print(f"{name}: complete, skipping")
            continue
        if hours_left() * 60 < reserve_min + 30:
            print(f"{name}: {hours_left():.1f} h left in this session, stopping here")
            break
        remote.mkdir(exist_ok=True)
        if not _claim(remote, hours_left()):
            print(f"{name}: claimed by another session, skipping")
            continue
        try:
            # 1. local working copy: restore a started replica, or seed a new one from the build
            if (remote / "_manifest.json").exists() and not (local / "_replica.json").exists():
                if not restore_from_drive(remote, local):
                    print("  continuing: produce() re-checks the trajectory against the checkpoint")
            local.mkdir(parents=True, exist_ok=True)
            meta_file = local / "_replica.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                if meta["build"] != build_id:
                    raise RuntimeError(f"{name} was started from another build ({meta['build']})")
            else:
                for f in (system, model, "build.json"):
                    if (build_dir / f).exists():
                        shutil.copy2(build_dir / f, local / f)

                meta = dict(replica=k, build=build_id, source=str(build_dir), ns=ns, temp=temp,
                            dt_fs=dt_fs, save_ps=save_ps, seed_equil=seed0 + k,
                            seed_prod=seed0 + 500 + k, ladder=equil.ladder, openmm=mm.__version__)
                
                meta_file.write_text(json.dumps(meta, indent=1))
            common = dict(temp=meta["temp"], dt_fs=meta["dt_fs"], save_ps=meta["save_ps"])

            # 2. equilibration can't resume mid-way, so one cut short is simply redone
            if not (local / equil_file).exists():
                print(f"{name}: equilibrating (seed {meta['seed_equil']})")
                equil.equilibrate(local, seed=meta["seed_equil"], **common)
                gc.collect()
                _mirror(local, remote)

            # 3. production resumes from its checkpoint, with the settings recorded for it
            print(f"{name}: production (seed {meta['seed_prod']})")
            sim = produce(local, save_name=save_name, ns=meta["ns"],
                          seed=meta["seed_prod"], max_hours=hours_left(),
                          reserve_min=reserve_min, mirror_to=remote,
                          mirror_every_min=mirror_every_min, **common)
            done = sim.currentStep >= int(meta["ns"] * 1e6 / meta["dt_fs"])
            del sim
            gc.collect()
            if not done:
                print(f"{name}: paused; run this cell again in the next session")
                break
            (remote / "_done.json").write_text(json.dumps(dict(by=SESSION, at=time.ctime())))
            print(f"{name}: complete")
            if cleanup_local:
                shutil.rmtree(local, ignore_errors=True)
        finally:
            _release(remote)