"""Trajectory analysis: per-run tables (analyze_run), fraying statistics (analyze_fraying)
and the spread across replicas (analyze_replicas). Column definitions are in the table
metrics document."""

from .constants import *
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial import cKDTree

def load_run(run_dir, traj, stride=1, model="model.cif"):
    """Load a trajectory without water, whether or not water was written.

    Parameters
    ----------
    run_dir : path-like
        Folder holding the trajectory and the built model.
    traj : str
        Trajectory name, with or without .xtc/.dcd ("md", "equil", "md.xtc").
    stride : int
        Keep every n-th frame.
    model : str
        Built model file in run_dir, used as the topology.

    Returns
    -------
    (md.Trajectory, md.Trajectory)
        The water-free trajectory and the built model restricted to the same atoms.

    """

    run_dir = Path(run_dir)
    path = run_dir / (traj if Path(traj).suffix in (".xtc", ".dcd") else f"{traj}.xtc")
    built = md.load(str(run_dir / model))
    top = built.topology
    dry = [a.index for a in top.atoms if not a.residue.is_water]
    with md.open(str(path)) as fh:
        n_file = fh.read(n_frames=1)[0].shape[1]
    if n_file == top.n_atoms:
        t = md.load(str(path), top=top, atom_indices=dry, stride=stride)
    elif n_file == len(dry):
        t = md.load(str(path), top=top.subset(dry), stride=stride)
    else:
        raise ValueError(f"{path.name} has {n_file} atoms, {model} has {top.n_atoms} "
                         f"({len(dry)} without water): wrong run folder?")
    return t, built.atom_slice(dry)

def design_pairs(db, nts):
    """Turn a dot-bracket design into the set of designed base pairs.

    Parameters
    ----------
    db : str
        One character per nucleotide, strands in file order joined by "+" or "&":
        "(" and ")" pair, "." is unpaired, "[]", "{}" and "<>" mark pseudoknotted pairs.
        Example, a 12-nt overhang and a 12-bp duplex: "." * 12 + "(" * 12 + "+" + ")" * 12
    nts : list of md.Residue
        Nucleotides of the topology, in file order.

    Returns
    -------
    set of frozenset
        Pairs of 0-based topology residue indices.
    """

    segs = db.replace("&", "+").replace(" ", "").split("+")
    by_chain = [[r for r in nts if r.chain.index == c] for c in sorted({r.chain.index for r in nts})]
    if [len(s) for s in segs] != [len(c) for c in by_chain]:
        raise ValueError(f"design lengths {[len(s) for s in segs]} != strand lengths "
                         f"{[len(c) for c in by_chain]}")
    flat = [r for c in by_chain for r in c]
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    stacks, pairs = {o: [] for o in closing.values()}, set()
    for i, ch in enumerate("".join(segs)):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in closing:
            if not stacks[closing[ch]]:
                raise ValueError(f"unbalanced '{ch}' at position {i + 1}")
            pairs.add(frozenset((flat[stacks[closing[ch]].pop()].index, flat[i].index)))
        elif ch != ".":
            raise ValueError(f"unexpected character {ch!r} in the design")
    if any(stacks.values()):
        raise ValueError("unbalanced design: unclosed brackets")
    return pairs

def _open_runs(closed, need):
    """(open stretches lasting >= need frames, longest open stretch in frames)"""
    edges = np.diff(np.r_[0, (~closed).astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int((lengths >= need).sum()), int(lengths.max(initial=0))

def compare_runs(*paths):
    """Summaries side by side: pass run folders (all their summaries) or summary files."""
    files = []
    for p in map(Path, paths):
        files += sorted((p / "analysis").glob("*_summary.json")) if p.is_dir() else [p]
    return pd.DataFrame([json.loads(f.read_text()) for f in files]).set_index("run").T

def _helix_ends(native, top):
    """Native pairs with no stacked native neighbour on at least one side: helix ends and
    bulge or loop edges, i.e. the pairs that can fray."""
    def chain(r):
        return top.residue(r).chain.index if 0 <= r < top.n_residues else None
    ends = set()
    for key in native:
        i, j = sorted(key)
        if not all(chain(i + di) == chain(i) and chain(j + dj) == chain(j)
                   and frozenset((i + di, j + dj)) in native
                   for di, dj in ((-1, 1), (1, -1))):
            ends.add(key)
    return ends


def _base_atoms(res):
    """Heavy atoms of a nucleotide's base: no sugar (primed names), no phosphate."""
    return [a.index for a in res.atoms if a.element.symbol != "H" and "'" not in a.name
            and a.name not in ("P", "OP1", "OP2", "OP3")]


def _residue_contacts(t, query, arm_list, cut):
    """C[f, a, b]: a base atom of nucleotide query[a] is within `cut` nm of a heavy atom of
    arm nucleotide arm_list[b] in frame f. Raw coordinates: the run is unwrapped, so these
    are contacts within the molecule, not with its periodic images."""
    top = t.topology
    q = [_base_atoms(top.residue(i)) for i in query]
    h = [[a.index for a in top.residue(r).atoms if a.element.symbol != "H"] for r in arm_list]
    qa, ha = np.concatenate(q).astype(int), np.concatenate(h).astype(int)
    q_own = np.repeat(np.arange(len(q)), [len(x) for x in q])
    h_own = np.repeat(np.arange(len(h)), [len(x) for x in h])
    C = np.zeros((t.n_frames, len(q), len(h)), bool)
    for f in range(t.n_frames):
        hit = cKDTree(t.xyz[f, qa]).sparse_distance_matrix(
            cKDTree(t.xyz[f, ha]), cut, output_type="ndarray")
        C[f, q_own[hit["i"]], h_own[hit["j"]]] = True
    return C


def _arm_allowed(pair, arm_list, top, skip):
    """Positions in arm_list that count as arm for this pair. An arm nucleotide within `skip`
    positions of EITHER partner, on that partner's strand, stacks on or is bonded to the
    pair, so it is ignored for both bases. The old rule checked each base's own strand only,
    which let the overhang base stacked on the terminal pair count as docking."""
    ch = [top.residue(m).chain.index for m in pair]
    return np.array([b for b, r in enumerate(arm_list)
                     if all(top.residue(r).chain.index != c or abs(r - m) > skip
                            for m, c in zip(pair, ch))], dtype=int)

def analyze_run(run_dir, traj, design=None, stride=1, max_ns=None, cut=0.35,
                min_open_ps=100, tail=0.5, image_every=10, model="model.cif",
                arm=None, arm_cut=0.5, arm_skip=2, min_contact_ps=500):
    """Structure, pairing, flexibility, ion, box and arm metrics for one trajectory.

    Parameters
    ----------
    run_dir : path-like
        Run folder: a replica folder, or a build folder for single runs.
    traj : str
        Trajectory name ("md", "equil", "A3_P1_200_ns_3", ...).
    design : str, optional
        Dot-bracket target structure (design_pairs). Without it, native pairs are those
        formed in the built model (both H-bonds under cut + 0.05 nm), which can admit
        diagonal cross-strand contacts in a distorted model; pass it whenever it is known.
    stride : int
        Keep every n-th frame.
    max_ns : float, optional
        Analyse only the first max_ns ns; appended to the output names ("md_100ns_...").
    cut : float
        H-bond heavy-atom distance (nm) under which a pair counts as formed (both bonds).
    min_open_ps : float
        Shortest open stretch counted in open_events.
    tail : float
        Final fraction of the run used for the summary's _tail means and slopes.
    image_every : int
        Frames between solute-image distance checks (1 = every frame, slower).
    model : str
        Built model file: topology and in_model reference.
    arm : None, str or False
        None: nucleotides outside the native inter-strand duplex, which includes any
        intra-strand structure such as an aptamer domain; an mdtraj selection string
        (e.g. "chainid 0 and resSeq 1 to 12"); False: no arm analysis.
    arm_cut : float
        Contact distance (nm) between arm heavy atoms and a pair's base atoms.
    arm_skip : int
        Arm nucleotides within this many positions of either partner, on that partner's
        strand, are ignored for that pair (they are bonded to or stacked on it).
    min_contact_ps : float
        Least time with, and without, contact before occ_arm_on / occ_arm_off are given.

    Returns
    -------
    frames : pandas.DataFrame   one row per frame (q_native, rmsd_nm, rg_nm, a_form, ...)
    pairs : pandas.DataFrame    one row per native pair, pair present in the model, or
                                non-native pair formed at least 5% of the time
    nt : pandas.DataFrame       one row per nucleotide (rmsf_nm, paired, c3_endo)
    summary : dict              run-level values: counts, tail means, slopes, arm statistics

    Files written in `<run_dir>/analysis`
    -----------------------------------
    <stem>_frames.csv, <stem>_pairs.csv, <stem>_nt.csv, <stem>_summary.json, and
    <stem>_native_series.npz (per-frame H-bond distances of the native pairs, read by
    analyze_fraying).

    Interpreting the results
    ------------------------
    - Integrity:    
        - q_native near 1: the target structure holds. 
        - A falling q_inter together with a jump in com_XY_nm: the strands are separating. 
        
    - Weak spots: native_below_50pct and the "weak or absent" table point to pairs that do
                  not hold; native_missing_in_model > 0 means the starting structure was
                  already off-target there.
    - Pairs:     occupancy is the main figure; d1_A (about 2.9 A when intact) and occ_hb1
                 separate partial from full breaking. Repeated open_events at helix ends
                 are fraying; long openings inside a stem deserve a look in the trajectory.
    - Convergence:  <metric>_slope_per_ns near 0 means a plateau. A steady fall in q_native
                 or q_intra means structure is still being lost and the tail means are not
                 equilibrium values; run longer before comparing systems.
    - Geometry:     rmsd_core_nm of a few tenths of a nm is typical of an intact duplex;
                 values near or above 1 nm signal rearrangement. a_form around 0.05-0.1 is
                 B-DNA; a rise toward 1 signals an A-like transition. rmsf_nm separates the
                 rigid core from floppy tails and loops.
    - Ions:         cations_5A should plateau; a rising slope means the ion atmosphere is still
                 adjusting, often because the structure is.
    - Box:          min_image_nm must stay above the 1.0 nm cutoff; the summary keeps the
                 minimum over sampled frames. Below it the solute touched its own image:
                 rebuild with more padding.
    - Arm:          arm_contact is how often arm nucleotides touch a pair's bases. Contact and
                 opening are coupled both ways (open bases reach their neighbours; a docked
                 base can open a pair), so arm_r_internal and arm_occ_shift show
                 association, not cause. arm_r_internal is computed over native pairs that
                 are not helix ends, rests on few points and is dominated by any that stay
                 open; comparing constructs with and without the extra region is the
                 stronger test.
    """
    run_dir = Path(run_dir)
    t, ref = load_run(run_dir, traj, stride, model)
    dt_ps = float(np.median(np.diff(t.time))) if t.n_frames > 1 else 0.0

    if not dt_ps > 0:
        dt_ps = 10.0 * stride
        print(f"  no usable time stamps, assuming {dt_ps:g} ps per frame")

    if max_ns:
        t = t[: int(round(max_ns * 1000 / dt_ps)) + 1]
    n, top = t.n_frames, t.topology
    nts = [r for r in top.residues if r.name in NT_BASE]
    base = {r.index: NT_BASE[r.name] for r in nts}
    chains = sorted({r.chain.index for r in nts})
    ids = [getattr(top.chain(c), "chain_id", None) for c in chains]
    if None in ids or len(set(ids)) < len(ids):
        ids = [chr(65 + k) for k in range(len(chains))]
    cid = dict(zip(chains, ids))
    lab = {r.index: f"{cid[r.chain.index]}:{r.name}{r.resSeq}" for r in nts}
    atom = lambda r, name: next((a.index for a in r.atoms if a.name == name), None)

    # every complementary pair (Watson-Crick or G-T wobble) that isn't a near neighbour
    cands = []
    for r1, r2 in itertools.combinations(nts, 2):
        b = (base[r1.index], base[r2.index])
        if b in PAIR_HBONDS:
            pu, py = r1, r2
        elif b[::-1] in PAIR_HBONDS:
            pu, py, b = r2, r1, b[::-1]
        else:
            continue
        if r1.chain.index == r2.chain.index and abs(r1.index - r2.index) < 4:
            continue
        hb = [(atom(pu, x), atom(py, y)) for x, y in PAIR_HBONDS[b]]
        if None not in sum(hb, ()):
            cands.append((pu.index, py.index, "".join(b), hb))

    if not cands:
        raise ValueError("no complementary base pairs to track")

    keys = [frozenset(c[:2]) for c in cands]
    inter = np.array([top.residue(i).chain.index != top.residue(j).chain.index
                      for i, j, *_ in cands])
    hb_atoms = np.array([p for c in cands for p in c[3]])

    def formed(tr, c):       # formed = both H-bond heavy-atom distances < c
        d = md.compute_distances(tr, hb_atoms, periodic=False).reshape(tr.n_frames, len(cands), 2)
        return (d < c).all(2), d[:, :, 0]

    chunks = [formed(t[s:s + 2000], cut) for s in range(0, n, 2000)]
    F = np.concatenate([c[0] for c in chunks])
    D = np.concatenate([c[1] for c in chunks])
    in_model = formed(ref, cut + 0.05)[0][0]              # unrelaxed model: 0.5 A of slack
    if design:
        native = design_pairs(design, nts)
        untracked = native - set(keys)
        if untracked:
            print(f"  {len(untracked)} designed pair(s) aren't WC/wobble, not tracked: "
                  + ", ".join("-".join(lab[i] for i in sorted(p)) for p in untracked))
    else:
        native = {k for k, m in zip(keys, in_model) if m}
    nat = np.array([k in native for k in keys])
    ends = _helix_ends(native, top)                       # NEW: helix ends and bulge edges

    # per-frame metrics
    fr = pd.DataFrame({"time_ns": np.arange(n) * dt_ps / 1000})

    for col, mask in (("q_native", nat), ("q_inter", nat & inter), ("q_intra", nat & ~inter)):
        fr[col] = F[:, mask].mean(1) if mask.any() else np.nan

    hb1 = D < cut                        # old criterion: first H-bond only (N1-N3 for WC pairs)
    fr["q_native_hb1"] = hb1[:, nat].mean(1) if nat.any() else np.nan

    fr["nonnative_pairs"] = F[:, ~nat].sum(1)

    heavy = np.array([a.index for a in top.atoms
                      if a.residue.index in base and a.element.symbol != "H"])
    core_res = {i for k in native for i in k}
    core = np.array([a for a in heavy if top.atom(a).residue.index in core_res], dtype=int)
    fr["rmsd_nm"] = md.rmsd(t, t, 0, atom_indices=heavy)
    if len(core) > 2:
        fr["rmsd_core_nm"] = md.rmsd(t, t, 0, atom_indices=core)
    ts = t.atom_slice(heavy)             # superposed copy; t keeps its unwrapped coordinates
    ts.superpose(ts[0], 0, atom_indices=np.searchsorted(heavy, core) if len(core) > 2 else None)
    X = ts.xyz
    fr["rg_nm"] = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
    rmsf = np.sqrt(((X - X.mean(0)) ** 2).sum(2).mean(0))
    quads = {r.index: [atom(r, x) for x in ("C1'", "C2'", "C3'", "C4'")] for r in nts}
    quads = {i: q for i, q in quads.items() if None not in q}
    north = md.compute_dihedrals(t, list(quads.values()), periodic=False) > 0   # C3'-endo
    fr["a_form"] = north.mean(1)
    P = [a.index for a in top.atoms if a.name == "P" and a.residue.index in base]
    ions = [a.index for a in top.atoms if a.residue.name in CATIONS]

    if P and ions:       # minimum image: ions are unwrapped and wander across box faces
        fr["cations_5A"] = [len(x) for x in md.compute_neighbors(t, 0.5, P, ions, periodic=True)]

    for c in chains:
        rs = [r for r in nts if r.chain.index == c]
        fr[f"e2e_{cid[c]}_nm"] = md.compute_distances(
            t, [[atom(rs[0], "O5'"), atom(rs[-1], "O3'")]], periodic=False)[:, 0]

    centre = {c: t.xyz[:, [a for a in heavy if top.atom(a).residue.chain.index == c]].mean(1)
              for c in chains}

    for c1, c2 in itertools.combinations(chains, 2):
        fr[f"com_{cid[c1]}{cid[c2]}_nm"] = np.linalg.norm(centre[c1] - centre[c2], axis=1)

    mi = np.full(n, np.nan)              # closest approach of the solute to its own image

    if t.unitcell_vectors is not None:
        shifts = [s for s in itertools.product((-1, 0, 1), repeat=3) if s > (0, 0, 0)]
        for f in range(0, n, image_every):
            x, box = t.xyz[f, heavy], t.unitcell_vectors[f]
            tree = cKDTree(x)
            mi[f] = min(tree.query(x + np.dot(s, box))[0].min() for s in shifts)

    fr["min_image_nm"] = mi

    # CHANGED: contacts per (pair nucleotide, arm nucleotide), neighbour exclusion per PAIR
    kept = [k for k in range(len(cands)) if nat[k] or in_model[k] or F[:, k].mean() >= 0.05]
    if arm is False:
        arm_res = set()
    elif isinstance(arm, str):
        arm_res = {top.atom(a).residue.index for a in top.select(arm)} & set(base)
    else:
        duplex = {i for key, d in zip(keys, nat & inter) if d for i in key}
        arm_res = set(base) - duplex if duplex else set()
    contact = {}
    todo = [k for k in kept if arm_res and not (set(cands[k][:2]) & arm_res)]
    if todo:
        arm_list = sorted(arm_res)
        query = sorted({i for k in todo for i in cands[k][:2]})
        qpos = {i: a for a, i in enumerate(query)}
        C = _residue_contacts(t, query, arm_list, arm_cut)
        for k in todo:
            i, j = cands[k][:2]
            keep = _arm_allowed((i, j), arm_list, top, arm_skip)
            contact[k] = C[:, [qpos[i], qpos[j]]][:, :, keep].any((1, 2))
        nk = [k for k in todo if nat[k]]
        if nk:
            fr["arm_contacts"] = np.sum([contact[k] for k in nk], axis=0)
        print(f"  arm: {len(arm_res)} nt ({lab[min(arm_res)]} ... {lab[max(arm_res)]}), "
              f"neighbours of either partner within {arm_skip} nt ignored")

    # per-pair and per-nucleotide tables
    need = max(1, int(round(min_open_ps / dt_ps)))
    need_c = max(1, int(round(min_contact_ps / dt_ps)))              # NEW
    rows = []

    for k in kept:
        i, j, typ, _ = cands[k]
        ev, longest = _open_runs(F[:, k], need)
        row = dict(pair=f"{lab[i]}-{lab[j]}", type=typ, kind="inter" if inter[k] else "intra",
                   native=bool(nat[k]), end=keys[k] in ends, in_model=bool(in_model[k]),
                   occupancy=round(float(F[:, k].mean()), 4),
                   occ_hb1=round(float(hb1[:, k].mean()), 4),
                   d1_A=round(10 * float(D[:, k].mean()), 2), open_events=ev,
                   open_events_hb1=_open_runs(hb1[:, k], need)[0],
                   longest_open_ns=round(longest * dt_ps / 1000, 3))
        c = contact.get(k)
        if c is not None:                                             # CHANGED
            on = int(c.sum())
            row.update(arm_contact=round(on / n, 4), contact_frames=on,
                       occ_arm_on=round(float(F[c, k].mean()), 4) if on >= need_c else np.nan,
                       occ_arm_off=round(float(F[~c, k].mean()), 4) if n - on >= need_c else np.nan)
        rows.append(row)

    pairs = pd.DataFrame(rows, columns=["pair", "type", "kind", "native", "end", "in_model",
                                        "occupancy", "occ_hb1", "d1_A", "open_events",
                                        "open_events_hb1", "longest_open_ns", "arm_contact",
                                        "contact_frames", "occ_arm_on", "occ_arm_off"]
                         ).astype({"native": bool, "end": bool, "in_model": bool,
                                   "occupancy": float, "arm_contact": float,
                                   "contact_frames": float, "occ_arm_on": float,
                                   "occ_arm_off": float})

    res_of = np.array([top.atom(a).residue.index for a in heavy])
    c3 = dict(zip(quads, north.mean(0)))
    nt = pd.DataFrame([dict(
        nt=lab[r.index], chain=cid[r.chain.index], resSeq=r.resSeq, base=base[r.index],
        rmsf_nm=round(float(rmsf[res_of == r.index].mean()), 4),
        paired=round(float(F[:, [k for k, c in enumerate(cands) if r.index in c[:2]]].any(1).mean()), 4),
        c3_endo=round(float(c3.get(r.index, np.nan)), 4)) for r in nts])

    # summary: means and trends over the last `tail` of the window
    last = fr[fr.time_ns >= fr.time_ns.iloc[-1] * (1 - tail)]
    stem = Path(traj).stem + (f"_{max_ns:g}ns" if max_ns else "")
    summary = dict(run=f"{run_dir.name}/{stem}", length_ns=round(float(fr.time_ns.iloc[-1]), 3),
                   strands=len(chains), nucleotides=len(nts), native_pairs=int(nat.sum()),
                   native_inter=int((nat & inter).sum()),
                   native_missing_in_model=int((nat & ~in_model).sum()),
                   native_below_50pct=int((pairs.native & (pairs.occupancy < 0.5)).sum()),
                   min_image_nm=round(float(np.nanmin(mi)), 3) if np.isfinite(mi).any() else None)

    summary.update({f"{c}_tail": round(float(last[c].mean()), 4) for c in fr.columns[1:]})

    for c in ("q_native", "q_inter", "q_intra", "a_form", "cations_5A", "rmsd_core_nm"):
        if c in last and len(last) > 3 and last[c].notna().all():
            summary[f"{c}_slope_per_ns"] = round(float(np.polyfit(last.time_ns, last[c], 1)[0]), 4)

    if contact:                                                       
        sel = pairs.native & pairs.arm_contact.notna()
        inner = sel & ~pairs.end            # end pairs fray regardless: keep them out of r
        x, y = pairs.arm_contact[inner], pairs.occupancy[inner]
        shift = (pairs.occ_arm_on - pairs.occ_arm_off)[sel].dropna()
        ok = inner.sum() > 2 and x.std() > 1e-6 and y.std() > 1e-6
        summary.update(
            arm_nt=len(arm_res),
            arm_contact_mean=round(float(pairs.arm_contact[sel].mean()), 4) if sel.any() else None,
            arm_r_internal=round(float(np.corrcoef(x, y)[0, 1]), 3) if ok else None,
            arm_r_internal_n=int(inner.sum()),
            arm_occ_shift=round(float(shift.mean()), 4) if len(shift) else None,
            arm_occ_shift_n=int(len(shift)))

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    for name, df in (("frames", fr), ("pairs", pairs), ("nt", nt)):
        df.to_csv(out / f"{stem}_{name}.csv", index=False)
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

    nk_all = [k for k in range(len(cands)) if nat[k]]                 
    if nk_all:
        hbn = np.array([p for k in nk_all for p in cands[k][3]])
        dn = md.compute_distances(t, hbn, periodic=False).reshape(n, len(nk_all), 2)
        np.savez_compressed(out / f"{stem}_native_series.npz",
                            time_ns=fr.time_ns.to_numpy(), d_nm=dn.astype(np.float32), cut=cut,
                            label=np.array([f"{lab[cands[k][0]]}-{lab[cands[k][1]]}" for k in nk_all]),
                            res=np.array([sorted(cands[k][:2]) for k in nk_all]),
                            res_chain=np.array([r.chain.index for r in top.residues]),
                            end=np.array([keys[k] in ends for k in nk_all]))

    print(f"{summary['run']}: {summary['length_ns']} ns, {len(chains)} strand(s), "
          f"{summary['native_pairs']} native pairs ({summary['native_inter']} inter-strand)")
    print(f"means over the last {tail:.0%}:\n" + last.drop(columns="time_ns").mean().round(3).to_string())

    weak = pairs[pairs.native & ((pairs.occupancy < 0.5) | ~pairs.in_model)]
    if len(weak):
        print("\nnative pairs that are weak or absent from the model:\n" + weak.to_string(index=False))
    odd = pairs[~pairs.native & (pairs.occupancy >= 0.05)].nlargest(5, "occupancy")

    if len(odd):
        print("\nmost populated non-native pairs:\n" + odd.to_string(index=False))

    if contact:                                                       
        tab = pairs[pairs.native & pairs.arm_contact.notna()][
            ["pair", "end", "occupancy", "arm_contact", "contact_frames", "occ_arm_on", "occ_arm_off"]]
        print(f"\narm contact vs pair stability ({len(arm_res)} arm nt):\n" + tab.to_string(index=False))
        if summary["arm_r_internal"] is not None:
            print(f"  across internal pairs (n={summary['arm_r_internal_n']}): "
                  f"r(contact, occupancy) = {summary['arm_r_internal']:+.2f}")
        if summary["arm_occ_shift"] is not None:
            print(f"  within pairs with >= {min_contact_ps:g} ps both with and without contact "
                  f"(n={summary['arm_occ_shift_n']}): occupancy shift = {summary['arm_occ_shift']:+.3f}")

    return fr, pairs, nt, summary


def _two_state(d, lo, hi, full_open=False):
    """d: (frames, pairs, 2) H-bond distances in nm -> True where the pair is closed.
    Closed once both H-bonds are < lo; open once one of them (both, if full_open) is > hi;
    in the dead band between, a pair keeps the state it had."""
    worst, best = d.max(2), d.min(2)
    core = np.where(worst < lo, 1, np.where((best if full_open else worst) > hi, -1, 0))
    last = np.where(core != 0, np.arange(len(core))[:, None], 0)
    np.maximum.accumulate(last, axis=0, out=last)
    state = np.take_along_axis(core, last, axis=0)
    start = np.where(worst[0] < 0.5 * (lo + hi), 1, -1)       # frames before the first decision
    return np.where(state == 0, start, state) == 1


def _runs(mask):
    """(start, length) of every run of True in a 1-D boolean array."""
    e = np.diff(np.r_[0, mask.astype(int), 0])
    s = np.flatnonzero(e == 1)
    return s, np.flatnonzero(e == -1) - s


def _stems(res, res_chain):
    """Native pairs grouped into stems, runs of stacked pairs (i, j), (i+1, j-1), ...;
    each stem is a list of column indices ordered 5'->3' along the strand of i."""
    col = {(int(i), int(j)): p for p, (i, j) in enumerate(res)}
    same = lambda a, b: 0 <= a < len(res_chain) and res_chain[a] == res_chain[b]
    nxt = {p: col[(i + 1, j - 1)] for (i, j), p in col.items()
           if (i + 1, j - 1) in col and same(i + 1, i) and same(j - 1, j)}
    stems = []
    for p in sorted(set(col.values()) - set(nxt.values()), key=lambda p: tuple(res[p])):
        s = [p]
        while s[-1] in nxt:
            s.append(nxt[s[-1]])
        stems.append(s)
    return stems

# FRAYING ANALYSIS

def analyze_fraying(run_dir, traj, max_ns=None, lo=None, hi=0.45, full_open=False,
                    n_blocks=5, long_ps=100):
    """How often, how long and how deep the native pairs open, from the
    <stem>_native_series.npz of `analyze_run`.

    Parameters
    ----------
    run_dir, traj, max_ns
        As given to analyze_run; they locate the .npz.
    lo : float, optional
        Closing threshold (nm): a pair is closed once both H-bonds are below it; defaults
        to analyze_run's cut.
    hi : float
        Opening threshold (nm): a pair is open once one H-bond exceeds it, or both with
        full_open. Between lo and hi a pair keeps its previous state (dead band), so
        flicker around one cutoff is not counted as openings.
    full_open : bool
        False counts any H-bond break, partial openings included; True only full openings.
    n_blocks : int
        Blocks for the within-run standard error p_open_se.
    long_ps : float
        Shortest opening counted in long_openings.

    Returns
    -------
    pairs : pandas.DataFrame  one row per native pair: p_open, openings, rates, lifetimes
    ends : pandas.DataFrame   one row per stem end: how often and how deep it frays
    summary : dict            thresholds and totals

    Files written in `<run_dir>`/analysis
    -----------------------------------
    <stem>_fraying_pairs.csv, <stem>_fraying_ends.csv, <stem>_fraying_frames.csv (fraying
    depth per end and internal openings per stem, frame by frame, for plotting) and
    <stem>_fraying.json.

    Interpreting the results
    ------------------------
    - Frequency: openings_per_100ns and k_open_per_us (openings per microsecond of closed
                 time, i.e. 1 / mean closed lifetime) are the most reproducible numbers; their counting error is about 1 / sqrt(openings).

    - Duration: tau_open_ns is the mean open lifetime; median_open_ns and max_open_ns show whether a few long openings dominate. p_open is driven by those, so it varies most between replicas.
    
    - Criterion: the default (either H-bond) counts several times more events than
                 full_open=True, mostly single H-bonds breaking; compare constructs under the same criterion and say which one you report.
    
    - Depth:    p_frayed is how often an end is open; p_depth_ge2 and p_depth_ge3 show how
                far fraying spreads. Fraying that rarely passes the second pair is normal for an intact duplex.
    
    - Internal: p_internal_open should stay near 0: internal pairs open on millisecond
                timescales, so openings inside a stem are rare in MD and worth checking in the trajectory 
                (a sheared G-C pair keeping its untracked N2-O2 bond also reads as open).
    
    - Melting:  p_melted near 1 for a one-pair "stem" usually means a spurious native pair
                that never formed (pass the design); for a real stem, it is unfolded.
    
    - Mostly-open pairs: k_open then counts re-openings after brief closings; read p_open
                and tau_open_ns instead.
    """
    out = Path(run_dir) / "analysis"
    stem = Path(traj).stem + (f"_{max_ns:g}ns" if max_ns else "")
    z = np.load(out / f"{stem}_native_series.npz")
    t_ns, label = z["time_ns"], [str(s) for s in z["label"]]
    lo = float(z["cut"]) if lo is None else lo
    dt = float(np.median(np.diff(t_ns))) if len(t_ns) > 1 else 0.01
    T = len(t_ns) * dt
    closed = _two_state(z["d_nm"], lo, hi, full_open)
    opened = ~closed

    rows = []
    for p, name in enumerate(label):
        c, o = closed[:, p], opened[:, p]
        n_op, n_cl = int((c[:-1] & o[1:]).sum()), int((o[:-1] & c[1:]).sum())
        t_c, t_o = float(c.sum() * dt), float(o.sum() * dt)
        s, L = _runs(o)
        whole = L[(s > 0) & (s + L < len(o))]          # episodes seen opening and closing
        blk = [b.mean() for b in np.array_split(o, n_blocks)]
        rows.append(dict(
            pair=name, end=bool(z["end"][p]), p_open=round(float(o.mean()), 4),
            p_open_se=round(float(np.std(blk, ddof=1) / np.sqrt(n_blocks)), 4),
            openings=n_op, closings=n_cl, t_closed_ns=round(t_c, 3), t_open_ns=round(t_o, 3),
            openings_per_100ns=round(100 * n_op / T, 2),
            long_openings=int((L * dt * 1000 >= long_ps).sum()),
            k_open_per_us=round(1000 * n_op / t_c, 2) if t_c else np.nan,
            tau_open_ns=round(t_o / n_cl, 3) if n_cl else np.nan,
            median_open_ns=round(float(np.median(whole)) * dt, 3) if len(whole) else np.nan,
            max_open_ns=round(float(L.max(initial=0)) * dt, 3)))
    pairs = pd.DataFrame(rows)

    fr, ends = pd.DataFrame({"time_ns": t_ns}), []
    for s in _stems(z["res"], z["res_chain"]):
        O = opened[:, s]
        a = np.cumprod(O, axis=1).sum(1)              # open pairs in a row from the first pair
        b = np.cumprod(O[:, ::-1], axis=1).sum(1)     # ... and from the last
        melted = a == len(s)
        inner = np.where(melted, 0, O.sum(1) - a - b) # open pairs not connected to an end
        tag = f"{label[s[0]]} .. {label[s[-1]]}"
        fr[f"internal[{tag}]"] = inner
        for at, depth in ([(s[0], a)] if len(s) == 1 else [(s[0], a), (s[-1], b)]):
            frayed = depth > 0
            fr[f"depth[{label[at]}]"] = depth
            ends.append(dict(
                end=label[at], stem=tag, p_frayed=round(float(frayed.mean()), 4),
                p_depth_ge2=round(float((depth >= 2).mean()), 4),
                p_depth_ge3=round(float((depth >= 3).mean()), 4),
                mean_depth_frayed=round(float(depth[frayed].mean()), 2) if frayed.any() else 0.0,
                max_depth=int(depth.max()),
                frays_per_100ns=round(100 * int((~frayed[:-1] & frayed[1:]).sum()) / T, 2),
                p_internal_open=round(float((inner > 0).mean()), 4),
                p_melted=round(float(melted.mean()), 4)))
    ends = pd.DataFrame(ends)

    summary = dict(run=f"{Path(run_dir).name}/{stem}", lo_nm=lo, hi_nm=hi, full_open=full_open,
                   length_ns=round(T, 2),
                   openings_per_100ns_ends=round(float(pairs.openings_per_100ns[pairs.end].sum()), 2),
                   openings_per_100ns_internal=round(float(pairs.openings_per_100ns[~pairs.end].sum()), 2),
                   p_internal_open_max=round(float(ends.p_internal_open.max()), 4) if len(ends) else None,
                   p_melted_max=round(float(ends.p_melted.max()), 4) if len(ends) else None)
    for name, df in (("pairs", pairs), ("ends", ends), ("frames", fr)):
        df.to_csv(out / f"{stem}_fraying_{name}.csv", index=False)
    (out / f"{stem}_fraying.json").write_text(json.dumps(summary, indent=2))

    print(f"fraying, {summary['run']}: closed < {10 * lo:.1f} A, open > {10 * hi:.1f} A "
          f"({'both' if full_open else 'either'} H-bond)")
    print(pairs.drop(columns=["closings", "t_closed_ns", "t_open_ns"]).to_string(index=False))
    print("\n" + ends.to_string(index=False))
    return pairs, ends, summary

# REPLICA ANALYSIS

"""
    metas = [json.loads((p / "_replica.json").read_text()) for p in reps]
    if len({(m["build"], json.dumps(m.get("ladder"))) for m in metas}) > 1:
        raise ValueError("replicas differ in build or equilibration ladder; pool them separately")
"""


def analyze_replicas(build_dir, traj="md", design=None, remote_root=None, fray_kw=None, **kw):
    """Analyse every finished replica of a build and summarise the spread across them.

    Runs `analyze_run` and `analyze_fraying` on each replica folder holding `_done.json`, then
    pools the results.

    Parameters
    ----------
    build_dir : path-like
        Build folder; replicas are read from <build_dir>/replicas unless remote_root is set.
    traj : str
        Production file stem of the replicas (run_replicas' save_name).
    design : str, optional
        Dot-bracket design for analyze_run (recommended).
    remote_root : path-like, optional
        Folder holding the rXX replica folders.
    fray_kw : dict, optional
        Settings for analyze_fraying, e.g. dict(full_open=True).
    **kw
        Settings for analyze_run, e.g. image_every=1, max_ns=100.

    Returns
    -------
    pairs_out : pandas.DataFrame    per native pair (index: pair, end): <metric>_mean and
                                    <metric>_sem across replicas, plus pooled rates
                                    k_open_pooled_per_us and tau_open_pooled_ns
    ends_out : pandas.DataFrame     per helix end: <metric>_mean and <metric>_sem
    summary_out : pandas.DataFrame  per run-level metric (fraying ones prefixed fray_):
                                    mean, sem, min, max, n

    Files written in <replica root>/pooled
    --------------------------------------
    <traj>_pairs.csv, <traj>_ends.csv, <traj>_summary.csv, besides each replica's own
    analysis files.

    Interpreting the results
    ------------------------
    - SEM:      standard deviation across replicas / sqrt(n); with n = 3 it is itself\
                uncertain by about +/-50%, so read it as an order of magnitude.
    \n
    - Agreement: compare a pair's p_open_sem with the replicas' own p_open_se (within-run \
                 block errors in their fraying tables). A replica SEM several times larger \
                 means the replicas sit in different states (a frayed versus a closed end, \
                 say) and the mean describes none of them: report them separately, or start \
                 them from a corrected structure.
    \n           
    - Pooled: rates all openings over all closed time; steadier than the mean of per-replica \
              rates when events are few. A mean k_open is dominated by any replica in which the pair is mostly open.
    \n        
    - Worst case: for min_image_nm read the min column (the worst replica), not the mean.
    \n
    Assign the return value (pairs_out, ends_out, summary_out = ...) so a notebook does \
    not print every table twice.
    """

    root = Path(remote_root or Path(build_dir) / "replicas")
    reps = sorted(p for p in root.glob("r[0-9]*") if (p / "_done.json").exists())
    if len(reps) < 2:
        raise ValueError(f"{len(reps)} finished replica(s) in {root}; need at least 2")

    metas = [json.loads((p / "_replica.json").read_text()) for p in reps]
    if len({(m["build"], json.dumps(m.get("ladder"))) for m in metas}) > 1:
        raise ValueError("replicas differ in build or equilibration ladder; pool them separately")

    P, E, S = [], [], []
    for p in reps:
        print(f"\n {p.name} ")
        _, pairs, _, s = analyze_run(p, traj, design=design, **kw)
        fp, fe, fs = analyze_fraying(p, traj, kw.get("max_ns"), **(fray_kw or {}))
        P.append(pairs[pairs.native].drop(columns="end").merge(fp, on="pair").assign(rep=p.name))
        E.append(fe.assign(rep=p.name))
        S.append({**s, **{f"fray_{k}": v for k, v in fs.items()}, "rep": p.name})
    P, E = pd.concat(P), pd.concat(E)
    S = (pd.DataFrame(S).set_index("rep").apply(pd.to_numeric, errors="coerce")
         .select_dtypes("number").dropna(axis=1, how="all"))

    def spread(df, by, cols):                 # mean and standard error across replicas
        g = df.groupby(by, sort=False)[cols]
        sem = g.std(ddof=1) / np.sqrt(g.count())
        res = pd.concat([g.mean().add_suffix("_mean"), sem.add_suffix("_sem")], axis=1)
        return res[[f"{c}_{x}" for c in cols for x in ("mean", "sem")]].round(4)

    pairs_out = spread(P, ["pair", "end"], [c for c in (
        "occupancy", "p_open", "openings_per_100ns", "long_openings", "k_open_per_us",
        "tau_open_ns", "max_open_ns", "arm_contact", "occ_arm_on", "occ_arm_off") if c in P])
    # pooled rates: every replica's events over every replica's time at risk
    tot = P.groupby(["pair", "end"], sort=False)[["openings", "closings", "t_closed_ns",
                                                  "t_open_ns"]].sum()
    pairs_out["k_open_pooled_per_us"] = (1000 * tot.openings / tot.t_closed_ns).round(2)
    pairs_out["tau_open_pooled_ns"] = (tot.t_open_ns / tot.closings).round(3)
    pairs_out = pairs_out.replace([np.inf, -np.inf], np.nan)
    ends_out = spread(E, ["end"], [c for c in E if c not in ("end", "stem", "rep")])
    summary_out = pd.DataFrame({"mean": S.mean(), "sem": S.std(ddof=1) / np.sqrt(S.count()),
                                "min": S.min(), "max": S.max(), "n": S.count()}).round(4)

    out = root / "pooled"
    out.mkdir(exist_ok=True)
    for name, df in (("pairs", pairs_out), ("ends", ends_out), ("summary", summary_out)):
        df.to_csv(out / f"{traj}_{name}.csv")
    print(f"\n{len(reps)} replicas, mean and SEM across them:\n")
    print(pairs_out.filter(regex="^(occupancy|p_open|openings_per_100ns|k_open_pooled|"
                                 "tau_open_pooled)").to_string())
    print("\n" + ends_out.to_string())
    return pairs_out, ends_out, summary_out