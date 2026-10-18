from .constants import *
import openmm as mm
import openmm.app as app
import numpy as np
from openmm import unit, Vec3, OpenMMException
from .system_builder import load_model

RESTRAINT_GROUP = 1

class MinLog(mm.MinimizationReporter):
    """Logs the minimiser so a bad ladder rung is visible instead of silent.

    Watch two things:
      maxF  -- if it is still thousands of kJ/mol/nm at the end, the minimiser
               did not converge and the next rung will start from a strained
               structure.
      calls -- OpenMM's L-BFGS restarts internally and resets iteration to 0
               each time, so the last reported iteration is NOT the total.
               Counting calls is the only honest measure.
    """

    def __init__(self, every=200, label=""):
        super().__init__()
        self.every, self.label = every, label
        self.calls = 0
        self.first_E = self.last_E = self.last_F = None

    def report(self, iteration, x, grad, args):
        g = np.asarray(grad).reshape(-1, 3)
        self.last_F = float(np.linalg.norm(g, axis=1).max())
        # args["system energy"] is evaluated at geometries where the minimiser's
        # INTERNAL constraint restraints are still being tightened, so it jumps
        # upward at every L-BFGS restart. Useful as a trace, useless as a
        # start/end pair. Those are read from the context instead.
        self.last_E = float(args["system energy"])
        self.calls += 1
        if self.calls % self.every == 0:
            print(f"      {self.label} call {self.calls:>5}  E={self.last_E:>13.1f}  "
                  f"maxF={self.last_F:>9.1f}")
        return False


def ladder_report(ladder, temp=310.15):
    """Print sigma = sqrt(kT/k) for each rung BEFORE running anything.

    This is the check that catches a bad k, and it is analytic, no simulation
    needed. Restraint ENERGY does not catch it: a stiffer spring holds atoms
    closer to the reference, so its energy goes down, not up. Restraint energy
    detects a bad reference geometry; sigma detects excessive stiffness.
    """
    kT = 0.0083144621 * temp
    print("ladder:")
    for kind, k, ns in ladder:
        if k <= 0:
            print(f"  {kind:<5} k={k:>6}  {ns:>5} ns   unrestrained")
            continue
        s = (kT / k) ** 0.5 * 10
        flag = ("  !! frozen, below bond-length noise" if s < 0.05 else
                "  !! too tight to allow A->B" if s < 0.15 else "")
        print(f"  {kind:<5} k={k:>6}  {ns:>5} ns   sigma={s:.2f} A{flag}")


def _superimpose(x, ref):
    """Kabsch: remove rigid-body translation and rotation before measuring RMSD."""
    xc, rc = x.mean(0), ref.mean(0)
    H = (x - xc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return (x - xc) @ R.T + rc


def _expected_density(system, topology):
    """Equilibrium density of the BOX, not of pure water.

    DNA has a partial specific volume near 0.55 cm3/g (density ~1.8), so a
    nucleic-acid box settles above 1.00 g/mL. Checking against 1.000 flags a
    correct run as broken on every single rung.
    """
    if topology is None:
        return 1.00
    n_w = sum(1 for r in topology.residues() if r.name.strip() in ("HOH", "WAT"))
    if not n_w:
        return 1.00
    m_w = n_w * 18.015
    m_tot = sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(system.getNumParticles()))
    m_solute = m_tot - m_w
    NA = 6.02214076e23
    v = m_w / NA / 0.993 + m_solute / NA / 1.75      # water + solute volumes
    return m_tot / NA / v


def _temperature(sim, system):
    """Instantaneous temperature.

    Prefer the integrator's own method. The hand-rolled fallback must count
    only MASSIVE particles: 4-site waters (OPC, TIP4P-Ew) carry one massless
    virtual site each, and counting those as degrees of freedom understates T
    badly. With OPC it reports 211 K for a system that is really at 310 K.
    """
    integ = sim.context.getIntegrator()
    if hasattr(integ, "computeSystemTemperature"):
        return integ.computeSystemTemperature().value_in_unit(unit.kelvin)
    ke = sim.context.getState(getEnergy=True).getKineticEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)
    n_massive = sum(1 for i in range(system.getNumParticles())
                    if system.getParticleMass(i).value_in_unit(unit.dalton) > 0)
    dof = 3 * n_massive - system.getNumConstraints() - 3
    return 2 * ke / (dof * 0.0083144621)


def _report_dynamics(sim, kind, k, ns, ref, heavy, temp, system, topology=None):

    st = sim.context.getState(getPositions=True, getEnergy=True,
                              enforcePeriodicBox=False)
    xyz = np.array(st.getPositions().value_in_unit(unit.nanometer))
    cur = xyz[heavy]
    if k > 0:
        # restraints pin ABSOLUTE positions, so raw displacement is the right
        # measure; it is exactly what the restraint energy penalises.
        d = np.linalg.norm(cur - ref, axis=1)
        note = ""
    else:
        # unrestrained, the molecule translates and rotates freely. Raw
        # displacement then measures DIFFUSION, not conformational change
        d = np.linalg.norm(_superimpose(cur, ref) - ref, axis=1)
        note = " (superimposed)"
    T = _temperature(sim, system)
    box = st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    mass = sum(system.getParticleMass(i).value_in_unit(unit.dalton)
               for i in range(system.getNumParticles()))
    dens = mass / 6.02214076e23 / (box * 1e-21)
    line = (f"  {kind:<5} k={k:>6}  {ns:>5} ns   T={T:>6.1f} K  "
            f"rho={dens:.3f}  d_rms={d.mean()*10:.3f} A{note}  d_max={d.max()*10:.2f} A")
    if k > 0 and d.mean() * 10 < 0.5:
        line += "   (restraint is blocking A->B)"
    print(line)
    if abs(T - temp) > 15:
        print(f"        !! temperature off target by {T-temp:+.0f} K")
    if k == 0:
        target = _expected_density(system, topology)
        if abs(dens - target) > 0.03:
            print(f"        !! density {dens:.3f} g/cm3 vs expected ~{target:.3f}")


def _report_minimisation(sim, log, k, e_before, system):
    tot = sim.context.getState(getEnergy=True).getPotentialEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)
    print(f"  min  k={k:>5}  {e_before:>13,.0f} -> {tot:>13,.0f} kJ/mol in {log.calls} calls")
    print(f"        maxF {log.last_F:>9.1f} kJ/mol/nm"
          + ("   !! not converged, next rung starts strained" if log.last_F > 1000 else ""))
    if k > 0:
        # guard: an exact 0.0 means force group RESTRAINT_GROUP is EMPTY, i.e.
        # setForceGroup() was never called -- not that the solute did not move.
        if not any(f.getForceGroup() == RESTRAINT_GROUP
                   for f in system.getForces()):
            print(f"        !! no force in group {RESTRAINT_GROUP}: "
                  "setForceGroup() missing, restraint energy unreadable")
            return
        r = sim.context.getState(getEnergy=True,
                                 groups={RESTRAINT_GROUP}).getPotentialEnergy(
                                 ).value_in_unit(unit.kilojoule_per_mole)
        print(f"        restraint energy {r:>11.2f} kJ/mol"
              + ("   !! reference geometry fights the force field"
                 if r > 0.02 * abs(tot) else ""))


def _platform():
  #Platform used for the sims
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    for n in ("CUDA", "HIP", "OpenCL"):
        if n in names:
            return mm.Platform.getPlatformByName(n), {"Precision": "mixed"}
    return mm.Platform.getPlatformByName("CPU"), {}


def _add_barostat(system, temp, freq=100, bar_value=1):
    system.addForce(mm.MonteCarloBarostat(bar_value * unit.bar, temp * unit.kelvin, freq))


def _simulation(system, topology, temp, dt_fs, seed):
    #Fixed the friction coeff here. 
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1 / unit.picosecond,
                                        dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    integ.setConstraintTolerance(1e-6)
    try:
        return app.Simulation(topology, system, integ, *_platform()), integ
    except OpenMMException as e:
        if os.environ.get("MD_ALLOW_CPU") != "1":     # a silent CPU run crawls and takes the cores
            raise

        print("Error:",e,'-> proceeding on CPU')
        return app.Simulation(topology, system, integ, mm.Platform.getPlatformByName("CPU"), {}), integ



def _check(csv, since_ps=None, expect_density=1.00):
    """
    Use the logging csv to assess the quality of the structure after a ladder run
    """
    try:
        rows = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True, deletechars=" "))
    except (IndexError, ValueError):
        rows = np.array([])
    if rows.size < 4:
        print("  (log too short to assess density and drift)")
        return
    col = lambda s: next(c for c in rows.dtype.names if s in c)
    # Restrict to the final unrestrained rung: across the whole log the slope is
    # dominated by the heating ramp (-1.20M cold -> -0.89M at 310 K), which reads
    # as a spurious multi-thousand kJ/mol/ns "drift".
    if since_ps is not None:
        m = rows[col("Time")] >= since_ps
        if m.sum() >= 4:
            rows = rows[m]
    d = rows[col("Density")][len(rows) * 3 // 4:].mean()
    e, t = rows[col("Total")], rows[col("Time")]
    drift = np.polyfit(t, e, 1)[0] * 1000                    # least squares, not endpoints
    nb = min(8, max(2, len(t) // 4))
    sd = np.std([np.polyfit(t[b], e[b], 1)[0] * 1000
                 for b in np.array_split(np.arange(len(t)), nb) if len(b) > 2])
    lo, hi = (expect_density - 0.03, expect_density + 0.03)
    print(f"  density {d:.4f} g/cm3 (expect ~{expect_density:.3f}) "
          f"{'OK' if lo < d < hi else '!! off target'}")
    print(f"  drift {drift:+,.0f} +/- {sd:,.0f} kJ/mol/ns "
          f"{'OK' if abs(drift) < 2 * sd or abs(drift) < 500 else '!! significant'}")


class Equilibrator():
    def __init__(self, ladder=None):
        # (restraint k in kJ/mol/nm^2, nanoseconds). 0 ns = minimisation.

        self._DEFAULT_LADDER = [("min",  1000, 0),      # minimise, solute held: let water find the surface
                                ("min",  0,    0),      # minimise, everything free
                                ("heat", 1000, 0.2),    # NVT-ish, ramp 10 K -> temp
                                ("npt",  500,  0.3),    # density settles
                                ("npt",  0,    5)]     # free: ion atmosphere + scaffold conversion(A->B)

        self._ladder = self._DEFAULT_LADDER

        if ladder is not None:
            self.set_ladder(ladder)

    def reset(self):
        self._ladder = self._DEFAULT_LADDER
        print(self.ladder)

    @property
    def ladder(self):
        return self._ladder

    def set_ladder(self, ladder):
        if not isinstance(ladder, (tuple, list)):
            raise TypeError("Wrong format for setting a ladder")

        self._ladder = ladder

    
    def equilibrate(self, out, sys_name="system", model_name="model", equil_name="equil",
                    temp=310.15, dt_fs=4.0, seed=1,save_ps=10):
        out = Path(out)
        system, model = load_model(out, sys_name, model_name)
        topology, positions = model.topology, model.positions

        heavy = [a.index for a in topology.atoms()
                if a.residue.name.strip() not in SOLVENT
                and a.element is not None and a.element.symbol != "H"]

        f = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
        f.addGlobalParameter("k", 0.0)
        for p in ("x0", "y0", "z0"):
            f.addPerParticleParameter(p)
        for i in heavy:
            f.addParticle(i, positions[i].value_in_unit(unit.nanometer))
        f.setForceGroup(RESTRAINT_GROUP)
        system.addForce(f)
        _add_barostat(system, temp)

        sim, integ = _simulation(system, topology, temp, dt_fs, seed)
        sim.context.setPositions(positions)
        print(f"equilibrating: {len(heavy):,} restrained heavy atoms, "
            f"{sim.context.getPlatform().getName()}")
        ladder_report(self.ladder, temp)

        sim.reporters.append(app.StateDataReporter(
            str(out / f"{equil_name}.csv"), max(1, int(save_ps * 1000 / dt_fs)), step=True, time=True,
            totalEnergy=True, temperature=True, density=True, speed=True))

        ref = np.array([positions[i].value_in_unit(unit.nanometer) for i in heavy])
        heated, elapsed_ps, free_start_ps = False, 0.0, None
        for i, (kind, k, ns) in enumerate(self.ladder):
            sim.context.setParameter("k", float(k))
            if kind == "min":
                e_before = sim.context.getState(getEnergy=True).getPotentialEnergy(
                    ).value_in_unit(unit.kilojoule_per_mole)
                log = MinLog(label=f"k={k}")
                sim.minimizeEnergy(maxIterations=5000 if i == 0 else 20000, reporter=log)
                _report_minimisation(sim, log, k, e_before, system)
                continue
            nsteps = max(1, int(ns * 1e6 / dt_fs))
            if k == 0 and free_start_ps is None:
                free_start_ps = elapsed_ps
            if k == 0 and ns >= 1.0 and not any(
                    isinstance(r, (app.XTCReporter, app.DCDReporter)) for r in sim.reporters):
                # the free rung is the only one whose convergence you actually need
                # to verify; without frames there is nothing to check it against.
                sim.reporters.append(app.XTCReporter(
                    str(out / f"{equil_name}.xtc"), max(1, int(20000 / dt_fs)),
                    enforcePeriodicBox=False))
            if kind == "heat":
                sim.context.setVelocitiesToTemperature(10 * unit.kelvin, seed)
                for j in range(20):
                    integ.setTemperature((10 + (temp - 10) * (j + 1) / 20) * unit.kelvin)
                    sim.step(max(1, nsteps // 20))
                integ.setTemperature(temp * unit.kelvin)
                heated = True
            else:
                if not heated:
                    # no "heat" rung ran, so velocities are still zero: assign them
                    # rather than letting the thermostat ramp from 0 K uncontrolled.
                    print("        (no heat rung: assigning velocities at target T)")
                    sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
                    heated = True
                sim.step(nsteps)
            elapsed_ps += ns * 1000
            _report_dynamics(sim, kind, k, ns, ref, heavy, temp, system, topology)

        sim.saveState(str(out / f"{equil_name}.xml"))

        # The equilibrated coordinates live in equil.xml, and
        # equil_final.cif below is a human-viewable copy of them.
        with (out / f"{equil_name}_final.cif").open("w") as fh:
            app.PDBxFile.writeFile(topology, sim.context.getState(
                getPositions=True, enforcePeriodicBox=False).getPositions(), fh, keepIds=True)
        _check(out / f"{equil_name}.csv", since_ps=free_start_ps,
            expect_density=_expected_density(system, topology))
        return sim