import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

st.set_page_config(page_title="ネジ・ナットボルトカウントアプリ", layout="wide")
st.title("🔩 ネジ・ナット・ボルト カウントアプリ")

# モデル読み込み
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"モデルファイル({model_path})が見つかりません。")
        st.stop()
    return YOLO(model_path)

# 画像処理＆描画共通関数
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

# タブで切り替え
tab1, tab2 = st.tabs(["🔩 ネジカウント", "🔧 ナットとボルトカウント"])

with tab1:
    st.header("ネジカウントアプリ")
    screw_model = load_model("screw_model.pt")
    uploaded_screw = st.file_uploader("ネジの画像をアップロード", type=["jpg", "jpeg", "png"], key="screw")
    if uploaded_screw:
        image = Image.open(uploaded_screw).convert("RGB")
        processed_image, counts = detect_and_draw(image, screw_model)
        st.image(processed_image, caption=f"検出ネジ数：{sum(counts.values())}本", use_container_width=True)

with tab2:
    st.header("ナット・ボルトカウントアプリ")
    nutbolt_model = load_model("nut_bolt_model.pt")
    uploaded_nutbolt = st.file_uploader("ナットまたはボルトの画像をアップロード", type=["jpg", "jpeg", "png"], key="nutbolt")
    if uploaded_nutbolt:
        image = Image.open(uploaded_nutbolt).convert("RGB")
        processed_image, counts = detect_and_draw(image, nutbolt_model)
streamlit run screw_counter_app.py
streamlit run screw_counter_app.py
        count_summary = "、".join([f"{k}: {v}個" for k, v in counts.items()])
        st.image(processed_image, caption=f"検出結果：{count_summary}", use_container_width=True)
