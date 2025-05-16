#!/bin/bash

# 创建输出文件夹
output_folder="form_rdk_rdkit_m5so_wb97x-d"
mkdir -p $output_folder

# 读取 CSV 文件并提取 SMILES 和 ID
while IFS=',' read -r id smiles; do
    echo "Processing ID: $id with SMILES: $smiles"
    python psi4_smiles.py "$smiles" "$id" "$output_folder"
done < <(tail -n +2 psi4_M5_SO.csv)  # 忽略第一行表头


