#!/usr/bin/env python3
# 20230407苏禹铭编写用于提取multiwfn的分子描述符
# Modified by Jianyong Yuan
import datetime
import os
import re
import subprocess
import time

import pandas as pd

def call_Multiwfn(all_target_files):
    result_file_list = []
    total_num_files = len(all_target_files)
    assert total_num_files > 0, "No target file was found!"
    print(f'Total {total_num_files} \'*{os.path.splitext(all_target_files[0])[1]}\' files were found.')
    print('Using Multiwfn to calculate various descriptors...')

    input_stream_command = [0, 100, 21, 'size', 0, 'MPP', 'a', 'n', 'q', 0, 
                            300, 5, 0, 8, 8, 1, 'h-1', 'h', 'l', 'l+1', 0, -10, 
                            12, 0, -1, 2, 2, 0, -1, 2, 4, 0, -1, -1, 'q']
    input_stream = "\n".join(list(map(str, input_stream_command)))
    for i, target_file in enumerate(all_target_files, start=1):
        view_bar(i, total_num_files)
        result_file = os.path.splitext(target_file)[0] + ".txt"
        arg = f'Multiwfn_noGUI {target_file} << EOF > {result_file}\n{input_stream}\nEOF\n'
        #arg = f'Multiwfn {target_file} << EOF > {result_file}\n{input_stream}\nEOF\n'
        subprocess.run(arg, shell=True)
        result_file_list.append(result_file)
    print('\n')
    
    return result_file_list

def search(ini_path, target_file_name):
    target_list = []
    for root, dirs, files in os.walk(ini_path, topdown=True):
        for f in files:     
            if re.search(target_file_name, f):
                full_path = os.path.join(root, f)
                target_list.append(full_path)

    return target_list

def data_extraction(file_list):
    data = []
    odi_values = []
    for file in file_list:
        with open(file, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        
        sample_name = os.path.basename(file)
        sample_name = os.path.splitext(sample_name)[0]
        
        pattern = r'Orbital delocalization index:\s*([\d.]+)'
        odi_values = [float(value) for value in re.findall(pattern, content)]
        odi_homo_1, odi_homo, odi_lumo, odi_lumoadd1 = odi_values[:4]
        odi_mean = sum(odi_values) / len(odi_values)
        odi_std = (sum((x - odi_mean) ** 2 for x in odi_values) / len(odi_values)) ** 0.5

        for line in lines:
            if ' Atoms:  ' in line:
                new_str = line.split(': ')[1]
                new_str = new_str.split(',')[0]
                atom_num = int(new_str)
            elif ' Molecule weight:      ' in line:
                new_str = line.split(': ')[1]
                new_str = new_str.split(' Da')[0]
                weight = float(new_str)
            elif 'Orbital' in line and 'HOMO' in line:
                homo = float(re.search(r'energy:\s+([\d.-]+)', line).group(1))
                homo_number = int(re.search(r'Orbital\s+(\d+)', line).group(1))
            elif 'Orbital' in line and 'LUMO' in line:
                lumo = float(re.search(r'energy:\s+([\d.-]+)', line).group(1))
            elif 'HOMO-LUMO gap:' in line:
                homo_lumo_gap = float(re.search(r'gap:\s+([\d.-]+)', line).group(1))
            elif 'Farthest distance:' in line:
                farthest_distance = float(re.search(r'---\s+([\d.]+)', line).group(1))
            elif ' Radius of the system:' in line:
                mol_radius = float(re.search(r'system:\s+([\d.]+)', line).group(1))
            elif ' Length of the three sides:' in line:
                mol_size = list(map(float, re.findall(r'([\d.]+)', line)))
                sorted_mol_size = sorted(mol_size)
                mol_size_short = sorted_mol_size[0]
                mol_size_2 = sorted_mol_size[1]
                mol_size_l = sorted_mol_size[2]
                length_ratio = mol_size_l / sum(mol_size)
                len_div_diameter = mol_size_l / (2 * mol_radius)
            elif 'Molecular planarity parameter (MPP) is' in line:
                mpp = float(re.search(r'is\s+([\d.]+)', line).group(1))
            elif ' Span of deviation from plane (SDP) is' in line:
                sdp = float(re.search(r'is\s+([\d.]+)', line).group(1))   
            elif "Magnitude of dipole moment:" in line:
                dipole_moment = float(re.search(r'a.u.\s+([\d.]+)', line).group(1))
            elif "Magnitude: |Q_2|=" in line:
                quadrupole_moment = float(re.search(r'\|Q_2\|=\s+([\d.]+)', line).group(1))
            elif "Magnitude: |Q_3|=" in line:
                octopole_moment = float(re.search(r'\|Q_3\|=\s+([\d.]+)', line).group(1))

        partindex=0 
        for idx, line in enumerate(lines):
            if '       ================= Summary of surface analysis =================' in line:
                partindex+=1
                if partindex == 1:
                    start_idx = idx + 2
                    #print(start_idx)           
                    line = lines[start_idx]
                    new_str = line.split('Volume: ')[1]
                    new_str = new_str.split('Bohr^3')[0]
                    volume = float(new_str)
                    line = lines[start_idx+1]
                    new_str = line.split('(M/V):')[1]
                    new_str = new_str.split('g/cm^3')[0]
                    density = float(new_str)
                    line = lines[start_idx+2]
                    new_str = line.split(' Minimal value:  ')[1]
                    new_str = new_str.split('kcal/mol   Maximal')[0]
                    espmin = float(new_str)
                    new_str = line.split('Maximal value: ')[1]
                    new_str = new_str.split('kcal/mol')[0]
                    espmax = float(new_str)

                    line = lines[start_idx+3]
                    new_str = line.split('Overall surface area:')[1]
                    new_str = new_str.split('Bohr^2 ')[0]
                    overall_surf_area = float(new_str)
                    line = lines[start_idx+4]
                    new_str = line.split('area:')[1]
                    new_str = new_str.split('Bohr^2 ')[0]
                    pos_surf_area = float(new_str)
                    line = lines[start_idx+5]
                    new_str = line.split('area:')[1]
                    new_str = new_str.split('Bohr^2 ')[0]
                    neg_surf_area = float(new_str)
                    line = lines[start_idx+6]

                    new_str = line.split('value:')[1]
                    new_str = new_str.split('a.u. ')[0]
                    overall_ave = float(new_str)
                    line = lines[start_idx+7]
                    new_str = line.split('value:')[1]
                    new_str = new_str.split('a.u. ')[0]
                    pos_ave = float(new_str)
                    line = lines[start_idx+8]
                    new_str = line.split('value:')[1]
                    new_str = new_str.split('a.u.')[0]
                    neg_ave = float(new_str)

                    line = lines[start_idx+9]
                    new_str = line.split(':')[1]
                    new_str = new_str.split('a.u.')[0]
                    Over_var = float(new_str)
                    line = lines[start_idx+12]
                    new_str = line.split(':')[1]
                    nu = float(new_str)
                    line = lines[start_idx+14]
                    new_str = line.split(':')[1]
                    new_str = new_str.split('a.u.')[0]
                    Pi = float(new_str)
                    line = lines[start_idx+15]
                    new_str = line.split(':')[1]
                    new_str = new_str.split('eV')[0]
                    MPI = float(new_str)
                    line = lines[start_idx+16]
                    new_str = line.split('Angstrom^2  (')[1]
                    new_str = new_str.split('%)')[0]
                    nonpolar_area = float(new_str)
                    line = lines[start_idx+17]
                    new_str = line.split('Angstrom^2  (')[1]
                    new_str = new_str.split('%)')[0]
                    polar_area = float(new_str)

                elif partindex == 2:
                    start_idx = idx + 4
                    #print(start_idx)           
                    line = lines[start_idx]
                    new_str = line.split(' Minimal value:  ')[1]
                    new_str = new_str.split('eV,  ')[0]
                    aliemin = float(new_str)
                    new_str = line.split('Maximal value: ')[1]
                    new_str = new_str.split('eV')[0]
                    aliemax = float(new_str)
                    line = lines[start_idx+4]
                    new_str = line.split('value:')[1]
                    new_str = new_str.split('a.u. ')[0]
                    alie_ave = float(new_str)
                    line = lines[start_idx+5]
                    new_str = line.split(':')[1]
                    new_str = new_str.split('a.u.')[0]
                    alie_var = float(new_str)

                elif partindex == 3:
                    start_idx = idx + 4
                    #print(start_idx)           
                    line = lines[start_idx]
                    new_str = line.split(' Minimal value:  ')[1]
                    new_str = new_str.split('eV,  ')[0]
                    leamin = float(new_str)
                    new_str = line.split('Maximal value: ')[1]
                    new_str = new_str.split('eV')[0]
                    leamax = float(new_str)
                    line = lines[start_idx+4]
                    new_str = line.split('value:')[1]
                    new_str = new_str.split('a.u. ')[0]
                    lea_ave = float(new_str)
                    line = lines[start_idx+7]
                    new_str = line.split(':')[1]
                    new_str = new_str.split('a.u.')[0]
                    lea_var = float(new_str)

        data.append([sample_name, atom_num, weight, homo, homo_number, lumo, homo_lumo_gap, 
                     odi_homo_1, odi_homo, odi_lumo, odi_lumoadd1,odi_mean, odi_std, 
                     farthest_distance, mol_radius, mol_size_short, mol_size_2, mol_size_l, 
                     length_ratio, len_div_diameter, mpp, sdp, dipole_moment, quadrupole_moment,
                     octopole_moment, volume, density, espmin,espmax, 
                     overall_surf_area, pos_surf_area, neg_surf_area, overall_ave, 
                     pos_ave, neg_ave, Over_var, nu, Pi, MPI, nonpolar_area,
                     polar_area, aliemin, aliemax, alie_ave, alie_var, leamin, leamax, 
                     lea_ave, lea_var
                    ])
    
    df = pd.DataFrame(data, columns=['SampleName', 'AtomNum', 'Weight', 'HOMO', 'HOMO_number', 'LUMO', 'HOMO_LUMO_Gap',
                                    'ODI_HOMO_1', 'ODI_HOMO', 'ODI_LUMO', 'ODI_LUMO_Add1', 'ODI_Mean', 'ODI_Std', 
                                    'Farthest_Distance', 'Mol_Radius','Mol_Size_Short', 'Mol_Size_2', 'Mol_Size_L', 'Length_Ratio', 'Len_Div_Diameter', 'MPP', 'SDP', 'Dipole_Moment', 'Quadrupole_Moment',
                                    'Octopole_Moment', 'Volume', 'Density', 'ESPmin', 'ESPmax', 
                                    'Overall_Surface_Area', 'Pos_Surface_Area', 'Neg_Surface_Area', 'Overall_Average', 'Pos_Average', 'Neg_Average', 'Overall_Variance', 'Nu', 'Pi', 'MPI', 'Nonpolar_Area',
                                    'Polar_Area', 'ALIEmin', 'ALIEmax', 'ALIE_Ave', 'ALIE_Var', 'LEAmin', 'LEAmax', 'LEA_Ave', 'LEA_Var'])
    print(df)
    df.to_csv('Multiwfn_descriptors.csv', index=False)

    print('\nInfo: The results were saved in the following path:\n      {0}'.format(os.getcwd() + os.sep + 'Multiwfn_descriptors.csv'))

def view_bar(current, total):
    p = int(100*current/total)
    a = "#"*int(p/2)
    b = "-"*(50-int(p/2))
    print("\r[{:s}{:s}] {:5.2%} ({:6d} / {:6d})".format(a, b, current/total, current, total), end='')

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def descriptor_info():
    print('\n=================================   Descriptor Information   =================================')
    print('> Size:')
    print('    + AtomNum: Number of Atoms                     + Weight: Molecular Weight')
    print('\n> Frontier Orbitals:')
    print('    + HOMO: Highest Occupied Molecular Orbital     + HOMO_number: HOMO Number')
    print('    + LUMO: Lowest Unoccupied Molecular Orbital    + HOMO_LUMO_Gap: HOMO-LUMO Gap')
    print('\n> Orbital Delocalization Index (ODI):')
    print('    + ODI_HOMO_1: ODI of HOMO-1                    + ODI_HOMO: ODI of HOMO')
    print('    + ODI_HOMO: ODI of HOMO                        + ODI_LUMO: ODI of LUMO')
    print('    + ODI_LUMO_Add1: ODI of LUMO+1                 + ODI_Mean: Mean of ODI')
    print('    + ODI_Std: Standard Deviation of ODI')
    print('\n> Shape:')
    print('    + Farthest_Distance: Farthest Distance between Atoms')
    print('    + Mol_Radius: Molecular Radius                 + Mol_Size_Short: Shortest Molecular Size')
    print('    + Mol_Size_2: Medium Molecular Size            + Mol_Size_L: Longest Molecular Size')
    print('    + Length_Ratio: Length Ratio                   + Len_Div_Diameter: Length Divided by Diameter')
    print('    + MPP: Maximum Positive Potential              + SDP: Surface-Derived Polarization')
    print('\n> Moment:')
    print('    + Dipole_Moment: Dipole Moment                 + Quadrupole_Moment: Quadrupole Moment')
    print('    + Octopole_Moment: Octopole Moment')
    print('\n> Quantitative Analysis of Molecular Surface:')
    print('    + Volume: Molecular Volume                     + Density: Molecular Density')
    print('    + ESPmin: Minimum Electrostatic Potential      + ESPmax: Maximum Electrostatic Potential')
    print('    + Pos_Surface_Area: Positive ESP Surface Area  + Neg_Surface_Area: Negative ESP Surface Area')
    print('    + Overall_Surface_Area: Overall Surface Area   + Overall_Average: Overall ESP Average')
    print('    + Pos_Average: Positive ESP Average            + Neg_Average: Negative ESP Average')
    print('    + Overall_Variance: Overall ESP Variance (sigma^2_tot)')
    print('    + Nu: Balance of charges                       + Pi: Internal charge separation')
    print('    + Polar_Area: Polar Area                       + Nonpolar_Area: Nonpolar Area')
    print('    + MPI: Molecular Polarizability Index')
    print('    + ALIEmin: Minimum ALIE (Average Localized Ionization Energy)')
    print('    + ALIEmax: Maximum ALIE                        + ALIE_Ave: ALIE Average')
    print('    + ALIE_Var: ALIE Variance')
    print('    + LEAmin: Minimum LEA (Local Electron Affinity)')
    print('    + LEAmax: Maximum LEA                          + LEA_Ave: LEA Average')
    print('    + LEA_Var: LEA Variance')
    print('==============================================================================================')


if __name__ == '__main__':
    start_time = time.time()
    start_date = datetime.datetime.now()
    print('      ***  The \'MultiwfnMLhelper\' script (Linux) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))

    try:
        current_path = os.getcwd()
        target_file_name = r".*\.fchk$"
        all_target_files = search(current_path, target_file_name)
        result_file_list = call_Multiwfn(all_target_files)
        data_extraction(result_file_list)
        descriptor_info()
    except Exception as e:
        print(f'Error: {e}')
        print('Note that: (1) The "isilent= 1" MUST be set in the settings.ini of Multiwfn.')
        print('           (2) Multiwfn can be DIRECTLY called by using the "Multiwfn" command.')

    end_time = time.time()
    end_date = datetime.datetime.now()
    print('\n      ***  The \'MultiwfnMLhelper\' script (Linux) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
    total_running_time(end_time, start_time)
        