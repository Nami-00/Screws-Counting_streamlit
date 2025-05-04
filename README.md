# 🔩 ネジカウントアプリ（YOLOv8 × Streamlit）

このアプリは、画像からネジの個数を自動で検出・カウントするツールです。YOLOv8 を使って物体検出を行い、Streamlit でシンプルなWebインターフェースを提供しています。

![demo](https://github.com/あなたのユーザー名/screw-counter-app/assets/xxx/画像ID) <!-- デモ画像を貼るならここ -->

---

## 🚀 アプリの使い方

### ✅ Streamlit Cloud で公開中（例）

👉 [https://yourname-screw-counter-app.streamlit.app](https://yourname-screw-counter-app.streamlit.app)

1. ページを開いて画像（JPG/PNG）をアップロード
2. 自動的にネジを検出して、バウンディングボックスと信頼度を表示
3. 検出数が下に表示されます

---

## 🛠️ インストール方法（ローカル実行）

```bash
git clone https://github.com/あなたのユーザー名/screw-counter-app.git
cd screw-counter-app
pip install -r requirements.txt
streamlit run app.py
