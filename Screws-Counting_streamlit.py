import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# モデルの読み込み
MODEL_PATH = r"C:\Users\namih\venv\hamadaya\ScrewsCounting_yolov8\trained_model\weights\best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="ネジカウントアプリ", layout="centered")
st.title("🔩 ネジカウントアプリ")
st.caption("画像をアップロードすると、ネジの個数と信頼度付きで検出されます")

# ファイルアップロード UI
uploaded_file = st.file_uploader("画像をアップロード（JPG/PNG）", type=["jpg", "jpeg", "png"])

# 検出＋表示処理
if uploaded_file:
    # PIL形式で画像を読み込み
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # OpenCV形式へ変換（YOLO用）
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 推論
    results = model(img_cv)[0]

    # PILに変換して描画準備
    image_draw = Image.fromarray(img_array)
    draw = ImageDraw.Draw(image_draw)
    
    # Windowsフォントの例（なければ font=None）
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, size=20) if os.path.exists(font_path) else None

    # バウンディングボックス描画
    screw_count = 0
    for box in results.boxes:
        screw_count += 1
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results.names[cls]}: {int(conf * 100)}%"

        # 座標取得
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 枠描画
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        # ラベル描画
        if font:
            draw.text((x1, y1 - 25), label, fill="yellow", font=font)
        else:
            draw.text((x1, y1 - 25), label, fill="yellow")

    # 結果表示
    st.image(image_draw, caption=f"🧮 検出されたネジの数：{screw_count}本", use_container_width=True)