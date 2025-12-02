#!/bin/bash

# MagicFace 批次表情生成腳本
# 使用說明: ./generate_all_emotions.sh

# 設定環境變數
export PYTHONPATH=/user_data/sab/NTU/magicface:$PYTHONPATH

# 輸入圖像路徑
INPUT_IMAGE="./processed_images/b.png"
BG_IMAGE="./processed_images/b_bg.png"

# 檢查輸入文件是否存在
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "錯誤: 找不到輸入圖像 $INPUT_IMAGE"
    exit 1
fi

if [ ! -f "$BG_IMAGE" ]; then
    echo "錯誤: 找不到背景圖像 $BG_IMAGE"
    exit 1
fi

echo "========================================="
echo "  MagicFace 批次表情生成開始"
echo "========================================="
echo "輸入圖像: $INPUT_IMAGE"
echo "背景圖像: $BG_IMAGE"
echo ""

# 創建輸出目錄
mkdir -p output

# 表情列表（表情名稱:AU組合:強度值）
declare -A EMOTIONS
EMOTIONS["1_happy"]="AU12+AU6:8+6"
EMOTIONS["2_surprised"]="AU1+AU2+AU5+AU26:6+6+5+7"
EMOTIONS["3_angry"]="AU4+AU9:7+5"
EMOTIONS["4_sad"]="AU1+AU4+AU15:5+3+6"
EMOTIONS["5_disgusted"]="AU9+AU15+AU17:6+4+3"
EMOTIONS["6_fearful"]="AU1+AU2+AU5+AU20:5+5+6+5"
EMOTIONS["7_neutral_smile"]="AU12:4"
EMOTIONS["8_confused"]="AU4+AU1:4+2"

# 表情中文名稱對照
declare -A EMOTION_NAMES_ZH
EMOTION_NAMES_ZH["1_happy"]="快樂"
EMOTION_NAMES_ZH["2_surprised"]="驚訝"
EMOTION_NAMES_ZH["3_angry"]="生氣"
EMOTION_NAMES_ZH["4_sad"]="悲傷"
EMOTION_NAMES_ZH["5_disgusted"]="厭惡"
EMOTION_NAMES_ZH["6_fearful"]="恐懼"
EMOTION_NAMES_ZH["7_neutral_smile"]="微笑"
EMOTION_NAMES_ZH["8_confused"]="困惑"

# 計數器
total=${#EMOTIONS[@]}
current=0

# 批次生成
for emotion in $(echo "${!EMOTIONS[@]}" | tr ' ' '\n' | sort); do
    current=$((current + 1))
    IFS=':' read -r aus variations <<< "${EMOTIONS[$emotion]}"
    emotion_zh="${EMOTION_NAMES_ZH[$emotion]}"

    echo "[$current/$total] 正在生成 $emotion_zh ($emotion) 表情..."
    echo "  - AU組合: $aus"
    echo "  - 強度值: $variations"

    # 創建輸出目錄
    mkdir -p "./output/$emotion"

    # 執行推理
    python inference.py \
        --img_path "$INPUT_IMAGE" \
        --bg_path "$BG_IMAGE" \
        --au_test "$aus" \
        --AU_variation "$variations" \
        --saved_path "./output/$emotion" 2>&1 | grep -E "Loading|%|saved|Error" | tail -5

    if [ $? -eq 0 ]; then
        echo "  ✓ $emotion_zh 表情生成完成！"
    else
        echo "  ✗ $emotion_zh 表情生成失敗！"
    fi
    echo ""
done

echo "========================================="
echo "  所有表情生成完成！"
echo "========================================="
echo ""
echo "輸出結果："
ls -lh output/*/b.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "總共生成 $total 種表情"
