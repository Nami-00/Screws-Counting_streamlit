import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# モデルファイルの存在チェック
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error("モデルファイル(best.pt)が見つかりません。")
    st.stop()

# モデルロード
model = YOLO(MODEL_PATH)

st.title("🔩 ボルトカウントアプリ")
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = model(img_cv)
        boxes = results[0].boxes
        names = model.names

        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        font = ImageFont.load_default()

        screw_count = 0
        for box in boxes:
            screw_count += 1
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            label = f"{names[cls]}: {int(conf * 100)}%"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, max(y1 - 15, 0)), label, fill="yellow", font=font)

        st.image(image_draw, caption=f"検出ネジ数：{screw_count}本", use_container_width=True)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
