in_collab = True
import os
from pathlib import Path
import openmmforcefields
try:
    from google.colab import drive
except Exception:
    in_collab = False

#Some constants for finding paths withing our drive folder
input_pdb_ext = "sources/Z1_aptamer/Aptamere_Z1_ext_sans_linker_ARN_COMPLEXE_ADN_FINAL.pdb"
input_pdb_ori = "sources/Z1_aptamer/Aptamere_Z1_ori_sans_linker_ARN_COMPLEXE_ADN_FINAL.pdb"

A3_Z1_ori_pdb_ori = "sources/A3_aptamer_triggers/A3_Z1_ori_ini.pdb"
A3_Z1_30_nt_pdb_ori = "sources/A3_aptamer_triggers/A3_Z1_30_ini.pdb"
A3_Z1_24_nt_pdb_ori = "sources/A3_aptamer_triggers/A3_Z1_24_ini.pdb"
A3_P1_res_pdb_ori = "sources/A3_aptamer_triggers/A3_P1_ini.pdb"

results_folder = "results"
Z1_ori_res_folder = os.path.join(results_folder, "Z1_ori")
Z1_ext_res_folder = os.path.join(results_folder, "Z1_ext")

Z1_ori_res_folder_v3= os.path.join(results_folder, "Z1_ori_v3")
Z1_ext_res_folder_v3= os.path.join(results_folder, "Z1_ext_v3")

A3_Z1_ori_res_folder = os.path.join(results_folder, "A3_ori")
A3_Z1_30_nt_res_folder = os.path.join(results_folder, "A3_Z1_30_nt")
A3_Z1_24_nt_res_folder = os.path.join(results_folder, "A3_Z1_24_nt")
A3_P1_res_folder = os.path.join(results_folder, "A3_P1")
A3_P1_res_folder_v3 = os.path.join(results_folder, "A3_P1_v3")


#Path to OpenMM Forcefield file
FFXML = Path(openmmforcefields.__file__).parent / "ffxml" / "amber"

#DNA forcefields available to us
# to learn more please check: https://docs.openmm.org/7.3.0/userguide/application.html#amber14
DNA_FF = {"OL21": "DNA.OL21.xml", "OL15": "DNA.OL15.xml", "bsc1": "DNA.bsc1.xml"}

#Water models available to us (picked from refactoring Alexis' notebook)
# to learn more please check: https://docs.openmm.org/7.3.0/userguide/application.html#water-models
WATER = {"TIP3P": ("amber14/tip3p.xml", "tip3p"),
         "OPC":   ("amber14/opc.xml", "tip4pew")}



DNA_NAME = {}
for target, names in {"DA": "A DA RA ADE", "DC": "C DC RC CYT",
                      "DG": "G DG RG GUA", "DT": "T DT THY U RU URA"}.items():
    for n in names.split():
        DNA_NAME.update({n: target, n + "5": target, n + "3": target, n + "N": target})
PHOSPHATE = ("P", "OP1", "OP2", "OP3")


SOLVENT = {"HOH", "WAT", "NA", "CL", "K", "MG", "Na+", "Cl-", "K+"}

SOLVENT_FOR_TRAJ = {"HOH", "WAT"}


NT_BASE = {"DA": "A", "DG": "G", "DC": "C", "DT": "T", "A": "A", "G": "G", "C": "C", "U": "U"}

PAIR_HBONDS = {("G", "C"): (("N1", "N3"), ("O6", "N4")),     # (purine atom, pyrimidine atom)
               ("A", "T"): (("N1", "N3"), ("N6", "O4")),
               ("A", "U"): (("N1", "N3"), ("N6", "O4")),
               ("G", "T"): (("N1", "O2"), ("O6", "N3")),     # wobble
               ("G", "U"): (("N1", "O2"), ("O6", "N3"))}

CATIONS = {"NA", "K", "LI", "RB", "CS", "MG", "CA"}