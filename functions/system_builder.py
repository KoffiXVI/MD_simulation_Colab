#We import the libraries that we will use
from .constants import *
import json
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit, Vec3
import openmmforcefields
import io
from collections import Counter
from pdbfixer import PDBFixer
from scipy.spatial import cKDTree

from scipy.spatial.distance import pdist
import itertools


#We define a function to save and load our modeller and system
def save_model(system, modeller, out, sys_name="system", modeller_name="model"):
    """
    Save the generated model. PDBx is used over PDB as PDB serial field
    overflows past 99.999 atoms and a solvated box could cross that
    """

    out = Path(out)

    out.mkdir(parents=True, exist_ok=True)

    (out / f"{sys_name}.xml").write_text(mm.XmlSerializer.serialize(system))

    with (out / f"{modeller_name}.cif").open("w") as fh:
        app.PDBxFile.writeFile(modeller.topology, modeller.positions,
                               fh, keepIds=True)

    print(f"saved {modeller.topology.getNumAtoms():,} atoms -> {out}/{sys_name}.xml, {out}/{modeller_name}.cif")


def load_model(out, sys_name, modeller_name):
    """
    load the saved model
    """
    out = Path(out)
    system = mm.XmlSerializer.deserialize((out / f"{sys_name}.xml").read_text())
    cif = app.PDBxFile(str(out / f"{modeller_name}.cif"))

    modeller = app.Modeller(cif.topology, cif.positions)
    return system, modeller


def prepare_dna_pdb(pdb_in, pdb_out, rna_to_dna=True, expect_strands=None, sequences=None,
                    replace_modified=False):
    """Thin DNA-specific layer around PDBFixer: we decide what the strands are and strip
    5'-phosphates; PDBFixer repairs residues and atoms from its templates."""
    pdb_in = Path(pdb_in)
    if not pdb_in.is_file():
        raise FileNotFoundError(f"{pdb_in} (working directory {Path.cwd()})")

    src = (app.PDBxFile if pdb_in.suffix.lower() in (".cif", ".pdbx") else app.PDBFile)(str(pdb_in))
    xyz = src.positions.value_in_unit(unit.nanometer)
    log, strands, dropped = [], [], Counter()

    # 1. nucleotides and heavy atoms only; a new strand wherever the O3'-P link is
    #    missing or stretched (OpenMM would otherwise bond consecutive residues by name)
    for res in src.topology.residues():
        atoms = {a.name.strip().replace("*", "'"): (a.element, np.array(xyz[a.index]))
                 for a in res.atoms() if a.element is None or a.element.symbol not in ("H", "D")}
        if "C1'" not in atoms:
            dropped[res.name] += 1
            continue
        prev = strands[-1][-1] if strands else None
        same = prev is not None and prev["chain"] == res.chain.index
        linked = (same and "P" in atoms and "O3'" in prev["atoms"] and
                  np.linalg.norm(atoms["P"][1] - prev["atoms"]["O3'"][1]) < 0.25)
        if same and not linked:
            log.append(f"!! backbone break before {res.chain.id}:{res.name}{res.id}: new strand")
        r = dict(chain=res.chain.index, name=res.name, id=res.id, atoms=atoms)
        if linked:
            strands[-1].append(r)
        else:
            strands.append([r])
    if dropped:
        log.append(f"dropped non-nucleotide residues {dict(dropped)}")
    if not strands:
        raise ValueError(f"no nucleotides found in {pdb_in}")

    # 2. one chain per strand, 5'-phosphates removed, handed to PDBFixer
    top, pos = app.Topology(), []
    for k, s in enumerate(strands):
        chain = top.addChain(chr(65 + k))
        for i, r in enumerate(s):
            res = top.addResidue(r["name"], chain, r["id"])
            for n, (el, x) in r["atoms"].items():
                if i == 0 and n in PHOSPHATE:
                    continue
                top.addAtom(n, el or app.element.Element.getBySymbol(n[0]), res)
                pos.append(Vec3(*x))
        if "P" in s[0]["atoms"]:
            log.append(f"removed 5'-terminal phosphate of strand {chr(65 + k)}")
    buf = io.StringIO()
    app.PDBFile.writeFile(top, unit.Quantity(pos, unit.nanometer), buf, keepIds=True)
    fixer = PDBFixer(pdbfile=io.StringIO(buf.getvalue()))

    # 3. PDBFixer: optional parent substitution for modified bases; every residue is then
    #    "mutated" to its DNA name, which deletes whatever the DNA template lacks (O2', OP3,
    #    misnamed atoms); missing heavy atoms (C7 after U->T, anything the model lost) are
    #    rebuilt from the template with a local minimisation
    if replace_modified:
        fixer.findNonstandardResidues()
        for r, new in fixer.nonstandardResidues:
            log.append(f"!! modified {r.name}{r.id} replaced by its parent {new}")
        fixer.replaceNonstandardResidues()
    residues = list(fixer.topology.residues())
    unknown = [f"{r.chain.id}:{r.name}{r.id}" for r in residues if r.name not in DNA_NAME]

    if unknown:
        raise ValueError(f"unrecognised or modified nucleotides {unknown}: this force field only "
                         f"has A/C/G/T (replace_modified=True substitutes the parent nucleotide)")

    rna = [r for r in residues if r.name.lstrip("R").startswith("U")
           or any(a.name == "O2'" for a in r.atoms())]
    if rna and not rna_to_dna:
        raise ValueError(f"{len(rna)} RNA residues (O2' or U); pass rna_to_dna=True to convert them")

    muts = {c.id: [f"{r.name}-{r.id}-{DNA_NAME[r.name]}" for r in c.residues()]
            for c in fixer.topology.chains()}

    for chain_id, m in muts.items():
        fixer.applyMutations(m, chain_id)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    rebuilt = Counter(a.name for atoms in fixer.missingAtoms.values() for a in atoms)
    fixer.addMissingAtoms()

    if rna:
        log.append(f"converted {len(rna)} RNA residues to DNA")
    if rebuilt:
        log.append(f"PDBFixer rebuilt missing heavy atoms {dict(rebuilt)}")

    # 4. 5' ends again in case this PDBFixer version re-adds phosphates, then sanity checks
    m = app.Modeller(fixer.topology, fixer.positions)
    m.delete([a for c in m.topology.chains() for a in next(c.residues()).atoms()
              if a.name in PHOSPHATE])
    seqs = ["".join(r.name[-1] for r in c.residues()) for c in m.topology.chains()]
    for c, q in zip(m.topology.chains(), seqs):
        rs = list(c.residues())
        log.append(f"strand {c.id}: 5'-{q}-3' ({len(q)} nt, residues {rs[0].id}-{rs[-1].id})")
    if expect_strands is not None and len(seqs) != expect_strands:
        raise ValueError(f"{len(seqs)} strand(s) found, {expect_strands} expected:\n  "
                         + "\n  ".join(log))
    if sequences is not None and [q.upper().replace("U", "T") for q in sequences] != seqs:
        raise ValueError(f"model sequences {seqs} differ from the expected {list(sequences)}")
    n_close = len(cKDTree(np.array(m.positions.value_in_unit(unit.nanometer))).query_pairs(0.10))
    if n_close:
        log.append(f"!! {n_close} heavy-atom pairs closer than 1 A: clashes in the model")
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(m.topology, m.positions, fh, keepIds=True)

    return Path(pdb_out), log

def check_box(build_dir, cutoff=1.0, model_name="model.cif"):
    """Worst-case solute-image gap if the solute rotates rigidly: shortest periodic image
    distance minus the solute's largest atom-atom distance. Leave room above the cutoff
    for a floppy overhang, which can stretch beyond its starting size."""
    cif = app.PDBxFile(str(Path(build_dir) / model_name))
    X = np.array(cif.positions.value_in_unit(unit.nanometer))
    solute = [a.index for a in cif.topology.atoms()
              if a.residue.name.strip() not in SOLVENT and a.element is not None]
    diam = pdist(X[solute]).max()
    V = np.array(cif.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    image = min(np.linalg.norm(np.dot(s, V))
                for s in itertools.product((-1, 0, 1), repeat=3) if any(s))
    gap = image - diam
    print(f"solute {diam:.2f} nm across, shortest image distance {image:.2f} nm, "
          f"worst-case gap {gap:.2f} nm" + ("   !! below the cutoff" if gap < cutoff else ""))
    return gap


def build_system(pdb_file, out, dna_ff="OL21", water="OPC", salt=0.10, cation="Na+",
                 anion="Cl-", padding=1.5, fit_box=True, rna_to_dna=True,
                 expect_strands=None, sequences=None):

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean, log = prepare_dna_pdb(pdb_file, out / "input_clean.pdb", rna_to_dna,
                                 expect_strands, sequences)

    print("\n".join("  " + line for line in log))
    water_xml, model = WATER[water]
    ff = app.ForceField(str(FFXML / DNA_FF[dna_ff]), water_xml)

    pdb = app.PDBFile(str(clean))
    m = app.Modeller(pdb.topology, pdb.positions)
    m.addHydrogens(ff)

    bad = ff.getUnmatchedResidues(m.topology)
    if bad:   # safety net: every offending residue, before any time goes into solvation
        raise ValueError("no force-field template for:\n  " + "\n  ".join(
            f"{r.chain.id}:{r.name}{r.id} {sorted(a.name for a in r.atoms())}" for r in bad))


    # padding= gives a CUBE sized by the longest axis. For an elongated
    # nucleic acid that is ~2x the water. boxShape='dodecahedron' is to
    # be preffered; aligning to principal axes first is better still.
    if fit_box:
        X = np.array(m.positions.value_in_unit(unit.nanometer))
        Xc = X - X.mean(0)
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        if np.linalg.det(Vt) < 0:
            Vt[2] *= -1                                  # a rotation, never a mirror image
        Q = Xc @ Vt.T
        m.positions = unit.Quantity([Vec3(*(q - Q.min(0) + padding)) for q in Q], unit.nanometer)
        s = Q.max(0) - Q.min(0) + 2 * padding
        box = dict(boxVectors=[Vec3(s[0], 0, 0), Vec3(0, s[1], 0), Vec3(0, 0, s[2])] * unit.nanometer)
    else:
        box = dict(padding=padding * unit.nanometer, boxShape="dodecahedron")

    m.addSolvent(ff, model=model, positiveIon=cation, negativeIon=anion,
                 ionicStrength=salt * unit.molar, neutralize=True, **box)

    if model == "tip4pew":
        m.addExtraParticles(ff)

    system = ff.createSystem(m.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, hydrogenMass=1.5 * unit.amu)

    save_model(system, m, out)

    
    box_nm = m.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    (out / "build.json").write_text(json.dumps(dict(
        source=str(Path(pdb_file).resolve()), openmm=mm.__version__, dna_ff=dna_ff,
        water=water, salt_M=salt, cation=cation, anion=anion, padding_nm=padding,
        fit_box=fit_box, box_nm=[round(v, 3) for v in box_nm],
        n_atoms=m.topology.getNumAtoms(), solute_image_gap=check_box(out), log=log), indent=2))

    return system, m