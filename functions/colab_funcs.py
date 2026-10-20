"""Drive helpers: mount and working directory, and the atomic copy, sync and restore that
mirror local runs to Drive (Drive itself cannot keep up with trajectory appends)."""

from .constants import *
import os
import json 
import shutil
import time

def set_dir(dest:str='MD_Simulation_Colab',mount_loc:str='/content/drive'):
    """Mount Google Drive (Colab only) and make `dest` the working directory.

    Parameters
    ----------
    dest : str
        Project folder, absolute or relative to the current directory; created if missing.

    mount_loc : str
        Drive mount point in Colab.

    Note
    ----
    In Colab: mounts Drive, creates `dest`, copies the sources folder in when
    `dest/sources` is missing. 
    """
    
    if not in_collab:
        return

    drive.mount(mount_loc)

    project_dir = os.path.join(mount_loc, "MyDrive", dest)
    sources_dir = os.path.join(project_dir, "sources")
    runtime_sources = "/content/MD_simulation_Colab/sources"

    os.makedirs(project_dir, exist_ok=True)

    if not os.path.exists(sources_dir):
        shutil.copytree(runtime_sources,sources_dir)
        print("created source directory in drive")

    os.chdir(project_dir)

    print("current working directy at:", os.getcwd())

#drive funcs

def restore_from_drive(remote, local):
    """Copy a mirrored run folder from Drive back to local disk and verify it.
    Returns True when every file in the manifest exists and has its recorded size, else False

    Parameters
    ----------
    remote : path-like
        Drive folder written by sync_dir; must hold _manifest.json.
    local : path-like
        Local working folder; created if missing; existing files are overwritten.

    Note
    ----
    False after a session that died mid-sync is usually harmless: each file is copied
    atomically, so it is either the old or the new version, and produce() re-checks the
    trajectory against the checkpoint before resuming. Read the printed problems first.
    """
    remote, local = Path(remote), Path(local)
    local.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((remote / "_manifest.json").read_text())
    bad = []
    for name, e in manifest.items():
        src, dst = remote / name, local / name
        if not src.exists():
            bad.append(f"{name}: missing on Drive"); continue
        shutil.copy2(src, dst)
        if dst.stat().st_size != e["size"]:
            bad.append(f"{name}: {dst.stat().st_size:,} bytes, expected {e['size']:,}")

    print("restore OK" if not bad else "!! restore problems:\n  " + "\n  ".join(bad))
    return not bad

def _copy_one(src: Path, dst: Path) -> int:
    """Copy one file through `<dst>.part` and a rename, so a kill mid-copy never leaves a
    truncated `dst`. Skipped when `dst` has the same size and is not older. Returns the
    bytes copied (0 when skipped); raises IOError on a short copy."""

    if not src.exists():
        return 0
    if dst.exists() and dst.stat().st_size == src.stat().st_size \
            and dst.stat().st_mtime >= src.stat().st_mtime:
        return 0
    tmp = dst.with_name(dst.name + ".part")

    shutil.copy2(src, tmp)

    try:                                         # push the bytes out of the page cache
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"short copy of {src.name}")

    os.replace(tmp, dst)
    return src.stat().st_size


STATE_LAST = (".chk", ".chk.bak", ".xml")
def sync_dir(local, remote, quiet=False) -> int:
    """Mirror the files of a local run folder to Drive (top level only, not recursive).

    Data files go first and restart state last (.chk, .chk.bak, .xml), so the checkpoint
    on Drive is never newer than the trajectory it belongs to; _manifest.json (file sizes)
    is written last, atomically. Returns Bytes copied by call; unchanged files are skipped.

    Parameters
    ----------
    local, remote : path-like
        Source and destination folders. remote's parent must exist (if not, Drive is
        probably not mounted); remote itself is created.
    quiet : bool
        Suppress the transfer summary line.
    """
    local, remote = Path(local), Path(remote)
    if not remote.parent.exists():
        raise FileNotFoundError(f"{remote.parent} does not exist; is Drive mounted?")
    remote.mkdir(parents=True, exist_ok=True)
    files = [p for p in local.iterdir()
             if p.is_file() and not p.name.endswith((".part", ".tmp"))]
    files.sort(key=lambda p: p.name.endswith(STATE_LAST))     # False (data) sorts first
    man = remote / "_manifest.json"
    manifest = json.loads(man.read_text()) if man.exists() else {}
    n, t0 = 0, time.time()
    for p in files:
        copied = _copy_one(p, remote / p.name)
        if copied:
            entry = {"size": p.stat().st_size}
            manifest[p.name] = entry
            n += copied

    tmp = man.with_name(man.name + ".part")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, man)                                     # manifest written last
    if n and not quiet:
        dt = time.time() - t0
        print(f"  [sync] {n/1e6:.1f} MB in {dt:.0f} s ({n/1e6/max(dt, 1e-3):.0f} MB/s) -> {remote}",
              flush=True)
    return n

def _mirror(local, remote):
    """sync_dir that never raises: a failed transfer is printed and the simulation goes on,
    since the local copy is intact and the next mirror retries."""
    try:
        sync_dir(local, remote)
    except Exception as e:
        print(f"  !! sync to {remote} failed: {type(e).__name__}: {e}")
        print(f"     the run is intact in {local}; copy it by hand")