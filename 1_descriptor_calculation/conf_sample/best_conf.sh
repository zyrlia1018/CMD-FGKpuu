#!/bin/bash

# 创建新文件夹 best_zyx
mkdir -p best_zyx

# 遍历当前文件夹下的所有以 LIT- 开头的文件夹
for dir in *; do
  # 如果文件夹内存在 crest_best.xyz 文件
  if [ -f "$dir/crest_best.xyz" ]; then
    # 将文件复制到 best_zyx 文件夹，并按文件夹名重命名
    cp "$dir/crest_best.xyz" "best_zyx/${dir}.xyz"
    echo "复制：$dir/crest_best.xyz 到 best_zyx/${dir}.xyz"
  fi
done

echo "文件复制完成！"


