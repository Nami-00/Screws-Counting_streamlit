import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import gdown

from ultralytics import YOLO

model = YOLO("best.pt")

st.title("🔩 ネジカウントアプリ")
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model(img_cv)[0]

    image_draw = Image.fromarray(img_array)
    draw = ImageDraw.Draw(image_draw)
    font = ImageFont.load_default()

    screw_count = 0
    for box in results.boxes:
        screw_count += 1
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results.names[cls]}: {int(conf * 100)}%"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, y1 - 15), label, fill="yellow", font=font)

    st.image(image_draw, caption=f"検出ネジ数：{screw_count}本", use_container_width=True)