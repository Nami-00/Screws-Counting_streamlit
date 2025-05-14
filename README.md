# 🔧 ナット・ボルト・ネジ カウントアプリ（YOLOv8 × Streamlit）

YOLOv8 を使って、画像から「ナット」「ボルト」「ネジ」を物体検出・カウントする Streamlit アプリです。  
**用途別に2つのタブに分かれており、デフォルトでは「ナット・ボルト」カウント機能が表示されます。**

## 🚀 公開アプリ
👉 [アプリを見る](https://screws-countingapp-8pbgad222ompbsshjqm6ge.streamlit.app/)

## 🎯 特長

- ✅ 画像をアップロードするだけで自動で検出・分類
- ✅ 「ナット」「ボルト」「ネジ」「ワッシャー」をクラスごとに個数表示
- ✅ 結果画像に枠線・信頼度（％）を表示
- ✅ 3つのタブで用途別にモデルを切り替え可能
- ✅ 合計個数を自動計算して表示

---

## 📸 使用イメージ（各タブの機能）

| タブ | 内容 | 使用データセット |
|------|------|------------------|
| 🔩 ナットとボルトカウント | ナットとボルトを検出・カウント | [Bolts and Nuts 2](https://universe.roboflow.com/mynewws/bolts-and-nuts-2/dataset/2) |
| 🔩 ネジカウント | ネジを一括検出・カウント | [Screws Counting](https://universe.roboflow.com/lfy/screws-counting/dataset/1) |
| 🔩 ナット・ボルト・ワッシャーカウント | 3種を分類してカウント | [Nuts Bolts Detection](https://universe.roboflow.com/yamaha-50qun/nuts-bolts-detection/dataset/1) |

---

## 🧠 使用技術

| 技術 | 内容 |
|------|------|
| モデル | YOLOv11（ultralytics） |
| UI | Streamlit |
| 画像処理 | Pillow + OpenCV |
| 重複除去 | torchvision.ops.nms を使用した IoUベースの後処理 |
| モデル形式 | `.pt`（PyTorch）形式の3モデルを使用 |


---

## 📁 フォルダ構成

screws-counter-app/
├── app.py # Streamlit アプリ本体
├── nut_bolt_model.pt # tab1用モデル
├── screw_model.pt # tab2用モデル
├── nut_bolt_washer_model.pt # tab3用モデル
├── requirements.txt # 必要ライブラリ一覧
└── README.md # このファイル

## 💻 ローカルでの実行方法

```bash
git clone https://github.com/Nami-00/screws-counter-app.git
cd screws-counter-app
pip install -r requirements.txt
streamlit run app.py
※ モデルファイルが大きい場合は .pt ファイルを gdown 経由でダウンロードする設計に変更可能です。
