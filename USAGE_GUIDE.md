# MagicFace 使用說明文件

## 簡介

MagicFace 是一個基於 Stable Diffusion 的面部表情編輯系統，使用動作單元（Action Units, AU）來精確控制面部表情變化。

## 系統需求

- Python 3.10+
- CUDA 12.8
- 至少 8GB GPU 記憶體
- 約 5GB 磁碟空間（用於模型文件）

## 安裝依賴

```bash
# 核心套件
pip install diffusers==0.25.1 transformers==4.30.0 einops imageio insightface
pip install onnxruntime-gpu

# 版本控制套件
pip install huggingface_hub==0.23.0 tokenizers==0.13.3
pip install accelerate==1.12.0 peft==0.10.0 bitsandbytes==0.48.2
pip install ml_dtypes==0.2.0
```

## 使用流程

### 步驟 1: 圖像預處理

將輸入圖像裁剪為 512x512 並檢測人臉：

```bash
python utils/preprocess.py \
  --img_path /path/to/your/image.jpg \
  --save_path ./processed_images/
```

**輸出**: `./processed_images/your_image.png`

### 步驟 2: 背景提取

提取背景並生成面部輪廓標記：

```bash
cd utils
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python retrieve_bg.py \
  --img_path ../processed_images/your_image.png \
  --save_path ../processed_images/
cd ..
```

**輸出**: `./processed_images/your_image_bg.png`

### 步驟 3: 面部表情編輯

使用 AU 參數生成新的面部表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/your_image.png \
  --bg_path ./processed_images/your_image_bg.png \
  --au_test 'AU12+AU6' \
  --AU_variation '5+3' \
  --saved_path ./output
```

**輸出**: `./output/your_image.png`

## 動作單元（AU）說明

MagicFace 支援 12 種動作單元，每個 AU 對應特定的面部肌肉動作：

| AU編號 | 肌肉動作 | 對應表情 | 強度範圍 |
|--------|---------|---------|---------|
| AU1 | 內眉提升 | 驚訝、擔憂 | -10 ~ +10 |
| AU2 | 外眉提升 | 驚訝 | -10 ~ +10 |
| AU4 | 眉頭緊皺 | 生氣、困惑 | -10 ~ +10 |
| AU5 | 上眼瞼提升 | 驚訝、恐懼 | -10 ~ +10 |
| AU6 | 臉頰提升 | 微笑 | -10 ~ +10 |
| AU9 | 皺鼻 | 厭惡 | -10 ~ +10 |
| AU12 | 嘴角上揚 | 快樂、微笑 | -10 ~ +10 |
| AU15 | 嘴角下拉 | 悲傷 | -10 ~ +10 |
| AU17 | 下巴提升 | 懷疑、思考 | -10 ~ +10 |
| AU20 | 嘴唇橫向伸展 | 恐懼 | -10 ~ +10 |
| AU25 | 嘴唇分開 | 放鬆、張嘴 | -10 ~ +10 |
| AU26 | 下顎下降 | 驚訝、張大嘴 | -10 ~ +10 |

## 表情範例

### 1. 快樂 (Happy)

最強的快樂表情，展現燦爛笑容：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU12+AU6' \
  --AU_variation '8+6' \
  --saved_path ./output/happy
```

**AU組合**:
- AU12 (嘴角上揚): +8
- AU6 (臉頰提升): +6

---

### 2. 驚訝 (Surprised)

張大眼睛和嘴巴的驚訝表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU1+AU2+AU5+AU26' \
  --AU_variation '6+6+5+7' \
  --saved_path ./output/surprised
```

**AU組合**:
- AU1 (內眉提升): +6
- AU2 (外眉提升): +6
- AU5 (上眼瞼提升): +5
- AU26 (下顎下降): +7

---

### 3. 生氣 (Angry)

眉頭緊皺、皺鼻的憤怒表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU4+AU9' \
  --AU_variation '7+5' \
  --saved_path ./output/angry
```

**AU組合**:
- AU4 (眉頭緊皺): +7
- AU9 (皺鼻): +5

---

### 4. 悲傷 (Sad)

眉毛內側上揚、嘴角下拉的悲傷表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU1+AU4+AU15' \
  --AU_variation '5+3+6' \
  --saved_path ./output/sad
```

**AU組合**:
- AU1 (內眉提升): +5
- AU4 (眉頭微皺): +3
- AU15 (嘴角下拉): +6

---

### 5. 厭惡 (Disgusted)

皺鼻、嘴角下拉的厭惡表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU9+AU15+AU17' \
  --AU_variation '6+4+3' \
  --saved_path ./output/disgusted
```

**AU組合**:
- AU9 (皺鼻): +6
- AU15 (嘴角下拉): +4
- AU17 (下巴提升): +3

---

### 6. 恐懼 (Fearful)

眉毛上揚、眼睛睜大、嘴唇橫向伸展：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU1+AU2+AU5+AU20' \
  --AU_variation '5+5+6+5' \
  --saved_path ./output/fearful
```

**AU組合**:
- AU1 (內眉提升): +5
- AU2 (外眉提升): +5
- AU5 (上眼瞼提升): +6
- AU20 (嘴唇橫向伸展): +5

---

### 7. 中性微笑 (Neutral Smile)

自然輕鬆的微笑：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU12' \
  --AU_variation '4' \
  --saved_path ./output/neutral_smile
```

**AU組合**:
- AU12 (嘴角上揚): +4

---

### 8. 困惑 (Confused)

眉頭微皺、內眉提升的困惑表情：

```bash
PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH python inference.py \
  --img_path ./processed_images/b.png \
  --bg_path ./processed_images/b_bg.png \
  --au_test 'AU4+AU1' \
  --AU_variation '4+2' \
  --saved_path ./output/confused
```

**AU組合**:
- AU4 (眉頭緊皺): +4
- AU1 (內眉提升): +2

## 批次處理腳本

如果要批次生成多個表情，可以使用以下腳本：

```bash
#!/bin/bash

# 設定環境變數
export PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH

# 輸入圖像路徑
INPUT_IMAGE="./processed_images/b.png"
BG_IMAGE="./processed_images/b_bg.png"

# 表情列表
declare -A EMOTIONS
EMOTIONS["happy"]="AU12+AU6:8+6"
EMOTIONS["surprised"]="AU1+AU2+AU5+AU26:6+6+5+7"
EMOTIONS["angry"]="AU4+AU9:7+5"
EMOTIONS["sad"]="AU1+AU4+AU15:5+3+6"
EMOTIONS["disgusted"]="AU9+AU15+AU17:6+4+3"
EMOTIONS["fearful"]="AU1+AU2+AU5+AU20:5+5+6+5"
EMOTIONS["neutral_smile"]="AU12:4"
EMOTIONS["confused"]="AU4+AU1:4+2"

# 批次生成
for emotion in "${!EMOTIONS[@]}"; do
    IFS=':' read -r aus variations <<< "${EMOTIONS[$emotion]}"
    echo "生成 $emotion 表情..."
    python inference.py \
        --img_path "$INPUT_IMAGE" \
        --bg_path "$BG_IMAGE" \
        --au_test "$aus" \
        --AU_variation "$variations" \
        --saved_path "./output/$emotion"
done

echo "所有表情生成完成！"
```

儲存為 `generate_all_emotions.sh` 並執行：

```bash
chmod +x generate_all_emotions.sh
./generate_all_emotions.sh
```

## 輸出結果

所有生成的圖像會保存在對應的輸出目錄：

```
output/
├── happy/b.png          # 快樂表情
├── surprised/b.png      # 驚訝表情
├── angry/b.png          # 生氣表情
├── sad/b.png            # 悲傷表情
├── disgusted/b.png      # 厭惡表情
├── fearful/b.png        # 恐懼表情
├── neutral_smile/b.png  # 中性微笑
└── confused/b.png       # 困惑表情
```

## 注意事項

1. **首次執行**會自動下載模型文件（約 4GB），需要較長時間
2. **GPU 記憶體**：每次推理需要約 6-8GB GPU 記憶體
3. **處理時間**：每張圖像約需 1-2 分鐘（首次執行需額外時間下載模型）
4. **圖像要求**：
   - 必須包含清晰的正面人臉
   - 建議解析度至少 512x512
   - 3D 卡通人物無法處理

## 常見問題

### Q: 如何調整表情強度？

A: 修改 `--AU_variation` 參數中的數值，範圍是 -10 到 +10。數值越大，表情越明顯。

### Q: 可以同時使用多少個 AU？

A: 建議同時使用 2-4 個 AU，過多的 AU 組合可能導致不自然的結果。

### Q: 如何創建自定義表情？

A: 參考 AU 說明表，選擇合適的 AU 組合，並調整強度參數。建議從較小的強度開始測試。

## 技術架構

MagicFace 使用以下技術：

- **Stable Diffusion**: 基礎生成模型
- **FACS (Facial Action Coding System)**: 面部動作編碼系統
- **VAE**: 變分自編碼器
- **UNet**: 擴散模型的去噪網絡
- **CLIP**: 文本編碼器

## 參考文獻

- 論文: [MagicFace: High-Fidelity Facial Expression Editing](論文連結)
- GitHub: [MagicFace Repository](https://github.com/...)
- HuggingFace 模型: [mengtingwei/magicface](https://huggingface.co/mengtingwei/magicface)

## 授權

請遵循原專案的授權協議。

---

## 📦 為什麼從 GitHub 克隆後有些檔案不見了？

### 不包含在 GitHub 儲存庫中的內容

為了控制儲存庫大小並符合 GitHub 限制，以下內容**不會上傳**到 GitHub：

#### 1. **大型模型文件**（總計約 10GB）

| 檔案/目錄 | 大小 | 為什麼不上傳 |
|-----------|------|-------------|
| `models--mengtingwei--magicface/` | ~6.5GB | 超過 GitHub 單檔 100MB 限制 |
| `models--sd-legacy--stable-diffusion-v1-5/` | ~800MB | 模型文件太大 |
| `denoising_unet/` | ~3.3GB | 模型權重文件 |
| `ID_enc/` | ~3.3GB | 模型權重文件 |
| `checkpoints/` | ~269MB | 檢查點文件 |
| `third_party_files/` | ~1.3GB | 第三方模型 |
| `79999_iter.pth` | ~51MB | 訓練權重 |

**解決方法**：按照「必要模型文件下載」章節從 HuggingFace 下載

#### 2. **輸出和處理過的圖像**

| 目錄 | 說明 |
|------|------|
| `output/` | 你生成的表情圖片會儲存在這裡 |
| `processed_images/` | 預處理後的輸入圖片 |
| `test_images/` | 你自己的測試圖片 |

**原因**：這些是使用者個人生成的內容，每個人都不同

**解決方法**：執行程式後會自動生成，或手動建立目錄：
```bash
mkdir -p output processed_images test_images
```

#### 3. **快取和臨時文件**

| 檔案/目錄 | 說明 |
|-----------|------|
| `.cache/`, `.locks/` | HuggingFace 模型快取 |
| `__pycache__/` | Python 編譯快取 |
| `*.log` | 執行日誌 |

**原因**：這些是執行時自動生成的臨時文件

**解決方法**：程式執行時會自動建立

#### 4. **IDE 和環境配置**

| 檔案/目錄 | 說明 |
|-----------|------|
| `.idea/`, `.vscode/` | IDE 個人設定 |
| `.env` | 環境變數（可能包含敏感資訊） |

**原因**：每個開發者的環境設定不同

### 克隆後的完整設定流程

```bash
# 1. 克隆儲存庫
git clone https://github.com/sab1126/magicface.git
cd magicface

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 下載模型（最重要！）
# 參見「必要模型文件下載」章節
pip install huggingface_hub
huggingface-cli download mengtingwei/magicface --local-dir models--mengtingwei--magicface
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models--sd-legacy--stable-diffusion-v1-5

# 4. 建立輸出目錄
mkdir -p output processed_images test_images

# 5. 開始使用！
# 將你的圖片放入專案目錄，然後執行預處理和推理
```

### 總結

- ✅ **程式碼和文檔**：完整包含在 GitHub 中
- ❌ **模型文件**：太大，需要從 HuggingFace 下載（約 10GB）
- ❌ **輸出圖片**：執行程式後才會生成
- ❌ **快取文件**：執行時自動建立

這樣的設計讓 GitHub 儲存庫保持輕量（僅 ~5MB），克隆速度更快！

---

**最後更新**: 2025-12-02
**版本**: 1.1
