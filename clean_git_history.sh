#!/bin/bash

# Git 歷史清理腳本
# 用於移除已經被追蹤的大型模型文件

echo "⚠️  警告：此腳本會修改 Git 歷史記錄！"
echo "在執行前請確保："
echo "1. 已備份所有重要文件"
echo "2. 所有團隊成員都知曉此變更"
echo "3. 準備好強制推送到遠端倉庫"
echo ""
read -p "確定要繼續嗎？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消操作"
    exit 0
fi

# 檢查是否安裝 git-filter-repo
if ! command -v git-filter-repo &> /dev/null; then
    echo "正在安裝 git-filter-repo..."
    pip install git-filter-repo
fi

echo "開始清理 Git 歷史..."

# 移除大型文件和目錄
git filter-repo --path models--mengtingwei--magicface --invert-paths --force
git filter-repo --path models--sd-legacy--stable-diffusion-v1-5 --invert-paths --force
git filter-repo --path denoising_unet --invert-paths --force
git filter-repo --path ID_enc --invert-paths --force
git filter-repo --path checkpoints --invert-paths --force
git filter-repo --path third_party_files --invert-paths --force
git filter-repo --path utils/third_party_files --invert-paths --force
git filter-repo --path utils/checkpoints --invert-paths --force
git filter-repo --path third_party --invert-paths --force
git filter-repo --path .locks --invert-paths --force
git filter-repo --path .cache --invert-paths --force

# 移除所有 .pth, .ckpt, .safetensors 文件
git filter-repo --path-glob '*.pth' --invert-paths --force
git filter-repo --path-glob '*.ckpt' --invert-paths --force
git filter-repo --path-glob '*.safetensors' --invert-paths --force
git filter-repo --path-glob '*.mat' --invert-paths --force
git filter-repo --path-glob '*.onnx' --invert-paths --force

# 移除輸出目錄
git filter-repo --path output --invert-paths --force
git filter-repo --path processed_images --invert-paths --force
git filter-repo --path test_images --invert-paths --force

echo ""
echo "✅ Git 歷史清理完成！"
echo ""
echo "接下來的步驟："
echo "1. 檢查清理結果: du -sh .git"
echo "2. 添加遠端倉庫: git remote add origin https://github.com/sab1126/magicface.git"
echo "3. 強制推送: git push --force --all"
echo "4. 強制推送標籤: git push --force --tags"
echo ""
echo "⚠️  注意：所有協作者需要重新克隆倉庫！"
