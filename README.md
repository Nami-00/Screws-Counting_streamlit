# 🔧 ナット・ボルト・ネジ カウントアプリ（YOLOv8 × Streamlit）

YOLOv8 を使って、画像から「ナット」「ボルト」「ネジ」を物体検出・カウントする Streamlit アプリです。  
**用途別に2つのタブに分かれており、デフォルトでは「ナット・ボルト」カウント機能が表示されます。**

## 🚀 公開アプリ
👉 [アプリを見る](https://screws-countingapp-8pbgad222ompbsshjqm6ge.streamlit.app/)

## 🎯 特長

- ✅ 画像をアップロードするだけで自動で検出・分類
- ✅ 「ナット」「ボルト」「ネジ」をクラスごとに個数表示
- ✅ 結果画像に枠線と信頼度（％）を表示
- ✅ タブ切り替えで2つの検出モデルを使い分け

## 📸 使用イメージ

| タブ | 内容 |
|------|------|
| 🔧 ナットとボルトカウント | ボルト、ナット、ワッシャーなど複数クラスを同時にカウント |
| 🔩 ネジカウント | 単一または類似形状のネジを一括検出・カウント |

## 🧠 使用技術
| 技術 | 内容 |
|------|------|
| モデル | YOLOv8（ultralytics） |
| UI | Streamlit |
| 画像処理 | Pillow + OpenCV |
| 学習済みモデル | `.pt`形式のPyTorchモデル（2種） |

---

## 📁 フォルダ構成

screws-counter-app/
├── app.py # Streamlit アプリ本体
├── screw_model.pt # ネジ検出モデル
├── nut_bolt_model.pt # ナット・ボルト検出モデル
├── requirements.txt # 必要ライブラリ
└── README.md # 本ファイル

## 💻 ローカルでの実行方法

```bash
git clone https://github.com/Nami-00/screws-counter-app.git
cd screws-counter-app
pip install -r requirements.txt
streamlit run app.py
※ モデルファイルが大きい場合は .pt ファイルを gdown 経由でダウンロードする設計に変更可能です。