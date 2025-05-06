import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
import cv2
import os
import tempfile
from torchvision.ops import nms
import torch

st.set_page_config(page_title="ネジ・ナットボルトカウントアプリ", layout="wide")
st.title("🔩 ネジ・ナット・ボルト カウントアプリ")

# モデル読み込み
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"モデルファイル({model_path})が見つかりません。")
        st.stop()
    return YOLO(model_path)

# 画像読み込み関数（拡張子を小文字に変換して読み込み）
def load_image(uploaded_file):
    try:
        # 拡張子を小文字に変換して安全に処理
        _, ext = os.path.splitext(uploaded_file.name)
        suffix = ext.lower() if ext else ".jpg"  # 万が一拡張子なしなら .jpg にする

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        return Image.open(tmp_path).convert("RGB")
    except UnidentifiedImageError:
        st.error("このファイルは画像として読み込めません。JPEGまたはPNG形式をご使用ください。")
        return None

# 画像処理＆描画共通関数
def detect_and_draw(image, model, conf_threshold=0.25, iou_threshold=0.4):
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 推論（confだけ適用、NMSは後処理で自前実行）
    results = model.predict(img_cv, conf=conf_threshold, iou=1.0)
    boxes_raw = results[0].boxes
    names = model.names

    # 必要なデータ抽出
    xyxy = boxes_raw.xyxy.cpu()
    scores = boxes_raw.conf.cpu()
    classes = boxes_raw.cls.cpu().int()

    # torchvisionのNMSで重複除去（IoUしきい値指定）
    keep = nms(xyxy, scores, iou_threshold=iou_threshold)
    xyxy = xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]

    # 描画処理
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    font = ImageFont.load_default()
    count_dict = {}

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        conf = scores[i].item()
        cls = classes[i].item()
        label = f"{names[cls]}: {int(conf * 100)}%"
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, max(y1 - 15, 0)), label, fill="yellow", font=font)
        count_dict[names[cls]] = count_dict.get(names[cls], 0) + 1

    return image_draw, count_dict

# ✅ タブの順序を「ナット・ボルト → ネジ」に変更
tab1, tab2 = st.tabs(["🔩 ナットとボルトカウント", "🔩 ネジカウント"])

with tab1:
    st.header("ナット・ボルトカウントアプリ")
    nutbolt_model = load_model("nut_bolt_model.pt")
    conf_threshold = st.slider("検出の信頼度しきい値（低いと検出が増えますが間違いも多くなります）", 0.0, 1.0, 0.25, 0.01, key="conf1")
    uploaded_nutbolt = st.file_uploader("ナットまたはボルトの画像をアップロード", type=None, key="nutbolt")
    if uploaded_nutbolt:
        image = load_image(uploaded_nutbolt)
        if image:
            processed_image, counts = detect_and_draw(image, nutbolt_model, conf_threshold, iou_threshold=0.1)
            count_summary = "、".join([f"{k}: {v}個" for k, v in counts.items()])
            total_count = sum(counts.values())
            st.image(processed_image, caption=f"検出結果（{conf_threshold:.2f}以上）：{count_summary}", use_container_width=True)
            st.markdown(f"### 🧮 合計個数：{total_count}個")
            
with tab2:
    st.header("ネジカウントアプリ")
    screw_model = load_model("screw_model.pt")
    conf_threshold = st.slider("検出の信頼度しきい値（低いと検出が増えますが間違いも多くなります）", 0.0, 1.0, 0.25, 0.01, key="conf2")
    uploaded_screw = st.file_uploader("ネジの画像をアップロード", type=None, key="screw")
    if uploaded_screw:
        image = load_image(uploaded_screw)
        if image:
            processed_image, counts = detect_and_draw(image, screw_model, conf_threshold, iou_threshold=0.1)
            st.image(processed_image, caption=f"検出ネジ数（{conf_threshold:.2f}以上）：{sum(counts.values())}本", use_container_width=True)
