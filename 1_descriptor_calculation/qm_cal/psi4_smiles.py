import psi4
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys

# 设置输出文件
psi4.core.set_output_file("output1.dat", True)

def mol2xyz(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    AllChem.UFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    string = "\n"
    for i, atom in enumerate(atoms):
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    string += "units angstrom\n"
    return string, mol


# 生成多个构象并找到最优构象
def mol2xyz_best_conformer(mol, num_confs=100):
    mol = Chem.AddHs(mol)  # 添加氢原子
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)  # 生成多个构象

    # UFF 力场优化每个构象，找到能量最低的构象
    energies = []
    for conf_id in range(num_confs):
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        ff.Minimize()
        energies.append((conf_id, ff.CalcEnergy()))

    # 找到能量最低的构象
    min_energy_conf = min(energies, key=lambda x: x[1])[0]

    # 生成最优构象的XYZ格式字符串
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    string = "{}\n\n".format(num_atoms)  # XYZ文件格式：原子数量 + 空行
    for atom in atoms:
        pos = mol.GetConformer(min_energy_conf).GetAtomPosition(atom.GetIdx())
        string += "{} {:.6f} {:.6f} {:.6f}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)  # 保留小数点后六位

    return string, mol

class MolecularProperties:
    def __init__(self, wfn):
        self.wfn = wfn

    @property
    def HOMO(self):
        return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha() - 1]

    @property
    def LUMO(self):
        return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha()]

def calculate_wfn(smiles, id, output_folder):
    fchk_filename = os.path.join(output_folder, f"{id}.fchk")

    if os.path.exists(fchk_filename):
        print(f"File {fchk_filename} already exists, skipping calculation for ID: {id}.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}. Skipping ID: {id}.")
        return
    # rdkit_single
    #xyz, mol = mol2xyz(mol)
    # rdkit_mutli
    xyz, mol = mol2xyz_best_conformer(mol)
    psi4.core.IOManager.shared_object().set_default_path("/home/zyrlia/zou/CMD-/CMD-ADMET/BBB/psi4/tmp")
    psi4.set_memory('4 GB')
    psi4.set_num_threads(8)

    try:
        psi_geom = psi4.geometry(xyz)
        scf_e, scf_wfn = psi4.energy("wb97x-d/def2-svp", return_wfn=True)
        psi4.fchk(scf_wfn, fchk_filename)

        mp = MolecularProperties(scf_wfn)

        print(f"Total Energy (ωB97X-D) for {id}:", scf_e)
        print(f"HOMO for {id}:", mp.HOMO)
        print(f"LUMO for {id}:", mp.LUMO)

    except Exception as e:
        print(f"Error calculating for ID: {id}. Error message: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_wfn.py <SMILES> <ID> <output_folder>")
        sys.exit(1)

    smiles = sys.argv[1]
    id = sys.argv[2]
    output_folder = sys.argv[3]

    calculate_wfn(smiles, id, output_folder)
