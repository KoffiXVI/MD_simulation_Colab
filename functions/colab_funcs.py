from .constants import *
import os
import json 
import shutil
import time

def set_dir(dest:str='MD_Simulation_Colab',mount_loc:str='/content/drive'):
    
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
    """New session: copy the run back to local disk and verify it against the manifest."""
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
    """Copy via a .part temp then rename, so a kill mid-copy cannot leave a
    truncated file at the destination, same reasoning as atomic checkpoints."""
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
    """Data files first, restart state last, then an atomically written manifest
    that a later session can verify against."""
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
    """Push to Drive. Never let a transfer failure kill a good simulation."""
    try:
        sync_dir(local, remote)
    except Exception as e:
        print(f"  !! sync to {remote} failed: {type(e).__name__}: {e}")
        print(f"     the run is intact in {local}; copy it by hand")