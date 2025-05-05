import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

st.set_page_config(page_title="ネジ・ナットボルトカウントアプリ", layout="wide")
st.markdown("<h1 style='font-size: 36px;'>🔩 ネジ・ナット・ボルト カウントアプリ</h1>", unsafe_allow_html=True)

# モデル読み込み
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"モデルファイル({model_path})が見つかりません。")
        st.stop()
    return YOLO(model_path)

# 共通処理関数
def detect_and_draw(image, model):
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model(img_cv)
    boxes = results[0].boxes
    names = model.names

    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    font = ImageFont.load_default()
    count_dict = {}

    for box in boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        label = f"{names[cls]}: {int(conf * 100)}%"
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, max(y1 - 15, 0)), label, fill="yellow", font=font)
        count_dict[names[cls]] = count_dict.get(names[cls], 0) + 1

    return image_draw, count_dict

# ✅ 順番入れ替え（左がナット・ボルト）
tab1, tab2 = st.tabs(["🔧 ナットとボルトカウント", "🔩 ネジカウント"])

with tab1:
    st.markdown("<h2 style='font-size:28px;'>🔧 ナット・ボルトカウントアプリ</h2>", unsafe_allow_html=True)
    nutbolt_model = load_model("nut_bolt_model.pt")
    uploaded_nutbolt = st.file_uploader("ナットまたはボルトの画像をアップロード", type=["jpg", "jpeg", "png"], key="nutbolt")
    if uploaded_nutbolt:
        image = Image.open(uploaded_nutbolt).convert("RGB")
        processed_image, counts = detect_and_draw(image, nutbolt_model)
        count_summary = "、".join([f"{k}: {v}個" for k, v in counts.items()])
        st.image(processed_image, caption=f"検出結果：{count_summary}", use_container_width=True)

with tab2:
    st.markdown("<h2 style='font-size:28px;'>🔩 ネジカウントアプリ</h2>", unsafe_allow_html=True)
    screw_model = load_model("screw_model.pt")
    uploaded_screw = st.file_uploader("ネジの画像をアップロード", type=["jpg", "jpeg", "png"], key="screw")
    if uploaded_screw:
        image = Image.open(uploaded_screw).convert("RGB")
        processed_image, counts = detect_and_draw(image, screw_model)
        st.image(processed_image, caption=f"検出ネジ数：{sum(counts.values())}本", use_container_width=True)

