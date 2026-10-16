import re
import urllib.parse

import gradio as gr
import joblib
import numpy as np

MODEL_PATH = "model_terbaik.joblib"
UNSAFE_CHARS = set('"<>`#{}|\\^')
DEFAULT_PATH = "/tienda1/publico/productos.jsp"

def wire_escape(s: str) -> str:
    out = []
    for ch in s:
        if ord(ch) <= 32 or ord(ch) >= 127 or ch in UNSAFE_CHARS:
            out.append(urllib.parse.quote(ch, safe="", encoding="utf-8"))
        else:
            out.append(ch)
    return "".join(out)

def fix_latin1(s: str) -> str:
    def one(m: re.Match) -> str:
        tok = m.group(0)
        raw = urllib.parse.unquote_to_bytes(tok)
        try:
            raw.decode("utf-8")
            return tok
        except UnicodeDecodeError:
            return urllib.parse.quote(raw.decode("latin-1"), safe="")

    return re.sub(r"(?:%[0-9A-Fa-f]{2})+", one, s)

def build_text(method: str, path: str, query: str, body: str) -> str:
    t = f"{method} {path}"
    if query:
        t += f"?{query}"
    if body:
        t += f" {body}"
    return t

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]
model_name = bundle["nama"]

def attack_score(X) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])
    margin = float(np.clip(model.decision_function(X)[0], -30, 30))
    return float(1.0 / (1.0 + np.exp(-margin)))

def predict_request(method: str, path: str, query: str, body: str):
    method = (method or "GET").upper().strip()
    path = (path or "").strip()
    query = (query or "").strip()
    body = (body or "").strip()
    if not (query or body):
        return {"Normal": 1.0, "Serangan": 0.0}, "Isi minimal salah satu: Query atau Body."

    path = wire_escape(fix_latin1(path)) or "/"
    query = wire_escape(fix_latin1(query))
    body = wire_escape(fix_latin1(body))
    text = build_text(method, path, query, body)

    X = vectorizer.transform([text])
    is_attack = int(model.predict(X)[0]) == 1
    score = attack_score(X)
    shown = score if is_attack else 1.0 - score
    label = {"Serangan": shown, "Normal": 1.0 - shown} if is_attack else {"Normal": shown, "Serangan": 1.0 - shown}
    return label, text

DESCRIPTION = (
    f"Model: **{model_name}** (TF-IDF). Dilatih pada CSIC 2010 + PayloadsAllTheThings "
    "(termasuk obfuskasi), hanya untuk SQL injection dan XSS.\n\n"
    "Skor adalah keluaran model, bukan probabilitas terkalibrasi (untuk SVM berupa sigmoid dari margin). "
    "Model dilatih pada trafik aplikasi toko CSIC (`/tienda1/...`), jadi path dan parameter di luar pola itu "
    "bisa kurang akurat. Ini prototipe penelitian, bukan pengganti WAF produksi."
)

EXAMPLES = [
    ["GET", DEFAULT_PATH, "search=Tito Febransyah", ""],
    ["GET", DEFAULT_PATH, "search=' OR 1=1 --", ""],
    ["GET", DEFAULT_PATH, "search=<script>alert(1)</script>", ""],
    ["GET", DEFAULT_PATH, "search=%27%20oR%201%3D1%20--", ""],
    ["POST", "/tienda1/publico/registro.jsp", "", "nombre=tito&email=a%40b.com"],
]

demo = gr.Interface(
    fn=predict_request,
    inputs=[
        gr.Dropdown(["GET", "POST", "PUT", "DELETE"], value="GET", label="Method"),
        gr.Textbox(value=DEFAULT_PATH, label="Path"),
        gr.Textbox(label="Query (tanpa tanda ?)", placeholder="search=..."),
        gr.Textbox(label="Body (form-encoded)", placeholder="nombre=...&email=..."),
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Hasil klasifikasi"),
        gr.Textbox(label="Teks yang dilihat model (setelah preprocessing)"),
    ],
    title="Deteksi SQLi dan XSS dengan Machine Learning",
    description=DESCRIPTION,
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
