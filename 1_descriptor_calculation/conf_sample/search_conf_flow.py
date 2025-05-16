import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 读取Excel文件
df = pd.read_excel('your_file.xlsx')

# 保存当前工作目录
original_dir = os.getcwd()

# 生成多个构象并找到最优构象
def mol2xyz_best_conformer(mol, num_confs=10):
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


# 处理每一个化合物
for index, row in df.iterrows():
    smiles = row['Compound_Structure']
    compound_id = row['ID']

    # 创建以化合物ID命名的文件夹
    folder_name = compound_id
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 转换SMILES为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法处理化合物 {compound_id}: SMILES 无效")
        continue

    # 将分子转换为最优构象的XYZ格式
    xyz, mol = mol2xyz_best_conformer(mol, num_confs=100)

    # 保存XYZ文件到对应文件夹
    xyz_file_path = os.path.join(folder_name, 'struc.xyz')
    with open(xyz_file_path, 'w') as xyz_file:
        xyz_file.write(xyz)

    # 切换到ID文件夹
    os.chdir(folder_name)

    # 检查是否已存在 crest.out 文件
    if not os.path.exists('crest.out'):
        crest_command = f"crest struc.xyz --gfn2 --gbsa h2o -T 16 > crest.out"
        subprocess.run(crest_command, shell=True, check=True)  # 等待CREST进程完成
        print(f"已处理并运行CREST命令：{compound_id}")
    else:
        print(f"已存在 crest.out 文件，跳过 CREST 运行：{compound_id}")

    # 运行完成后，切换回原始工作目录
    os.chdir(original_dir)


