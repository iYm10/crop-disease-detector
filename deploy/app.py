import html
import io
import base64
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE = Path(__file__).parent
IMG = 224

st.set_page_config(page_title="Crop Doctor", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------- text (EN / AR)
T = {
    "en": {
        "title": "Crop Doctor", "sub": "Offline AI that spots plant diseases from a single leaf photo, and tells you when it is not sure.",
        "lang": "Language", "settings": "Settings", "conf": "Minimum confidence", "conf_h": "Below this, the app says 'uncertain' instead of guessing.",
        "ent": "Maximum uncertainty (entropy)", "ent_h": "Measures prediction spread; does not reliably identify non-leaf images.",
        "tips": "📸 Tips for a good photo", "tips_l": ["One leaf, filling most of the frame", "Daylight, no harsh shadows", "Sharp focus, plain background if possible"],
        "tab1": "🔍 Diagnose", "tab2": "🕘 History", "tab3": "ℹ️ About",
        "src": "Image source", "up": "Upload", "cam": "Camera", "up_l": "Upload a leaf photo", "cam_l": "Take a photo of one leaf",
        "empty": "👆 Add a leaf photo to start.", "result": "Result", "top3": "Top-3 predictions", "details": "Technical details",
        "healthy": "Looks healthy", "disease": "Disease detected", "uncertain": "Not sure",
        "adv_d": "Confirm with an agricultural extension officer before applying any treatment.",
        "adv_u": "Retake the photo (one leaf, good light, fill the frame) or ask a local expert.",
        "why_c": "confidence is below your threshold", "why_e": "the prediction is too spread out across classes",
        "crop": "Crop", "cond": "Condition", "confl": "Confidence", "time": "Inference time",
        "hist_empty": "No diagnoses yet. Your session history will appear here.", "dl": "Download CSV", "clr": "Clear history",
        "m1": "Test accuracy", "m2": "Model size", "m3": "Answered photos", "m4": "Accuracy when answered",
        "how": "How it works",
        "how_t": "A MobileNetV2 network, fine-tuned on 54k PlantVillage leaf images (38 classes), is compressed to a 2.6 MB TensorFlow Lite model. "
                 "It runs on the CPU with no internet. When the top confidence is low or the probabilities are scattered, it abstains.",
        "lim": "Limitations",
        "lim_t": ["Trained on lab-style photos: accuracy on real field photos is likely lower.", "Covers 14 crops and 38 classes only.",
                  "Not a substitute for expert diagnosis, and gives no treatment advice."],
        "by": "Built by Yahya Majrashi",
    },
    "ar": {
        "title": "طبيب المحاصيل", "sub": "ذكاء اصطناعي يعمل بدون إنترنت يكتشف أمراض النباتات من صورة ورقة واحدة، ويخبرك إذا لم يكن متأكدًا.",
        "lang": "اللغة", "settings": "الإعدادات", "conf": "أقل نسبة ثقة", "conf_h": "تحت هذه النسبة يقول التطبيق «غير متأكد» بدل التخمين.",
        "ent": "أقصى درجة تشتت (entropy)", "ent_h": "يقيس تشتت التوقعات ولا يضمن اكتشاف الصور التي ليست أوراقًا.",
        "tips": "📸 نصائح لصورة جيدة", "tips_l": ["ورقة واحدة تملأ معظم الصورة", "إضاءة نهارية بدون ظلال قوية", "تركيز واضح وخلفية بسيطة إن أمكن"],
        "tab1": "🔍 التشخيص", "tab2": "🕘 السجل", "tab3": "ℹ️ حول التطبيق",
        "src": "مصدر الصورة", "up": "رفع", "cam": "الكاميرا", "up_l": "ارفع صورة ورقة", "cam_l": "صوّر ورقة واحدة",
        "empty": "👆 أضف صورة ورقة للبدء.", "result": "النتيجة", "top3": "أفضل 3 توقعات", "details": "تفاصيل تقنية",
        "healthy": "تبدو سليمة", "disease": "تم اكتشاف مرض", "uncertain": "غير متأكد",
        "adv_d": "أكّد التشخيص مع مرشد زراعي قبل استخدام أي علاج.",
        "adv_u": "أعد التصوير (ورقة واحدة وإضاءة جيدة وتملأ الصورة) أو اسأل خبيرًا محليًا.",
        "why_c": "الثقة أقل من الحد الذي حددته", "why_e": "التوقعات متشتتة بين عدة فئات",
        "crop": "المحصول", "cond": "الحالة", "confl": "الثقة", "time": "زمن التحليل",
        "hist_empty": "لا توجد تشخيصات بعد. سيظهر سجل هذه الجلسة هنا.", "dl": "تحميل CSV", "clr": "مسح السجل",
        "m1": "دقة الاختبار", "m2": "حجم النموذج", "m3": "الصور المُجاب عنها", "m4": "الدقة عند الإجابة",
        "how": "كيف يعمل",
        "how_t": "شبكة MobileNetV2 مدرّبة على 54 ألف صورة ورقة من PlantVillage (38 فئة)، ومضغوطة إلى نموذج TensorFlow Lite بحجم 2.6 ميجابايت. "
                 "تعمل على المعالج بدون إنترنت، وتمتنع عن الإجابة عندما تكون الثقة منخفضة أو التوقعات متشتتة.",
        "lim": "القيود",
        "lim_t": ["تدرّب على صور مخبرية: الدقة على صور الحقول الحقيقية غالبًا أقل.", "يغطي 14 محصولًا و38 فئة فقط.",
                  "ليس بديلًا عن تشخيص الخبير ولا يقدم نصائح علاج."],
        "by": "تطوير يحيى الماجرشي",
    },
}

# ---------------------------------------------------------------- model: original preprocessing, lazy resource loading
@st.cache_resource
def load():
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(BASE / 'model.tflite'))
    it.allocate_tensors()
    with open(BASE / 'labels.json', encoding='utf-8') as f:
        labels = json.load(f)
    return it, labels, threading.Lock()


def predict(img):
    it, labels, lock = load()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    x = np.asarray(img.convert('RGB').resize((IMG, IMG)), dtype='float32')[None]
    start = time.perf_counter()
    with lock:
        it.set_tensor(inp['index'], x)
        it.invoke()
        scores = it.get_tensor(out['index'])[0].copy()
    return scores, (time.perf_counter() - start) * 1000, labels


def pretty(name):
    crop, _, condition = name.partition('___')
    return crop.replace('_', ' ').strip(), condition.replace('_', ' ').strip()


def e(value):
    return html.escape(str(value))


for key, default in {'history': [], 'result': None, 'upload_key': 0, 'rotation': 0, 'lang': 'ar'}.items():
    if key not in st.session_state:
        st.session_state[key] = default

lang = st.session_state.lang

def tr(en, ar):
    return ar if lang == 'ar' else en

t = T[lang]

# A single visual system; all artwork and fonts work without external requests.
CSS = '''
<style>
:root {--paper:#f6f7f2;--panel:#ffffff;--ink:#203d31;--muted:#586a5c;--line:#dfe5d9;--green:#245b43;--lime:#d9edb5}
.stApp {background:var(--paper);color:var(--ink)}
[data-testid="stHeader"] {background:#f6f7f2e8}
.block-container {max-width:1320px;padding:2rem 2.2rem 3rem}
[data-testid="stSidebar"] {background:#ecf0e6;border-inline-end:1px solid var(--line)}
[data-testid="stSidebar"] {color:var(--ink);color-scheme:light}
[data-testid="stMarkdownContainer"],label,[data-testid="stWidgetLabel"],h1,h2,h3 {color:var(--ink)}
[data-testid="stCaptionContainer"] {color:var(--muted)}
h1,h2,h3 {letter-spacing:-.035em} h3 {font-size:1.18rem}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"] {background:#fff;color:var(--ink);border-color:var(--line);border-radius:12px}
input,textarea {color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}
[data-testid="stExpander"] {background:#fff;border-color:var(--line);border-radius:15px}
[data-testid="stFileUploaderDropzone"] {background:#f0f4e9;border:1px dashed #9ba98b;border-radius:18px;padding:25px 18px;color:var(--ink)}
[data-testid="stFileUploaderDropzone"] * {color:var(--ink)}
/* Explicit widget states keep native dark-theme defaults out of this light UI. */
.stApp {color-scheme:light}
:is(.stButton,.stDownloadButton,[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"]) button,
button[kind="secondary"],button[kind="tertiary"] {
    background:#fff!important;color:#203d31!important;border:1px solid #a7b7a2!important;
    border-radius:12px;min-height:44px;transition:background-color .15s,border-color .15s,box-shadow .15s;
}
:is(.stButton,.stDownloadButton,[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"]) button:hover:not(:disabled),
button[kind="secondary"]:hover:not(:disabled),button[kind="tertiary"]:hover:not(:disabled) {
    background:#e5efdc!important;color:#193e2b!important;border-color:#52734b!important;
    box-shadow:0 3px 10px #22492b10;
}
:is(.stButton,.stDownloadButton,[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"]) button:active:not(:disabled),
button[kind="secondary"]:active:not(:disabled),button[kind="tertiary"]:active:not(:disabled) {
    background:#d6e5ca!important;color:#193e2b!important;border-color:#245b43!important;
}
button[kind="primary"] {background:#245b43!important;border-color:#245b43!important;color:#fff!important;min-height:50px}
button[kind="primary"]:hover:not(:disabled) {background:#194832!important;border-color:#194832!important;color:#fff!important}
button[kind="primary"]:active:not(:disabled) {background:#123b28!important;color:#fff!important}
/* Streamlit wraps button labels in Markdown containers. Inherit the button color. */
button:is([kind="primary"],[kind="secondary"],[kind="tertiary"]) :is(p,span,[data-testid="stMarkdownContainer"]),
:is(.stButton,.stDownloadButton,[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"]) button :is(p,span,[data-testid="stMarkdownContainer"]) {
    color:inherit!important;-webkit-text-fill-color:currentColor!important;
}
button:focus-visible,[data-baseweb="tab"]:focus-visible,[role="slider"]:focus-visible {
    outline:3px solid #52734b!important;outline-offset:3px!important;
}
:is(.stButton,.stDownloadButton,[data-testid="stFileUploaderDropzone"],[data-testid="stCameraInput"]) button:disabled,
button[kind]:disabled {
    background:#e7ebe3!important;color:#65725f!important;border-color:#c4cebd!important;
    opacity:1!important;box-shadow:none!important;cursor:not-allowed;
}
/* Sidebar text, selection fields and their portal-mounted menus. */
[data-testid="stSidebar"] :is(label,[data-testid="stWidgetLabel"],[data-testid="stMarkdownContainer"]) {color:#203d31}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#586a5c}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"] {
    background:#fff!important;color:#203d31!important;border-color:#a7b7a2!important;
}
[data-baseweb="select"]>div:hover {background:#f3f7ed!important;border-color:#52734b!important}
[data-baseweb="select"]:focus-within,[data-baseweb="input"]:focus-within {
    outline:2px solid #52734b;outline-offset:2px;border-radius:12px;
}
[data-baseweb="select"] :is(span,input,svg) {color:#203d31!important}
input::placeholder,textarea::placeholder {color:#62735e!important;-webkit-text-fill-color:#62735e!important;opacity:1}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"],
[role="listbox"] {background:#fff!important;color:#203d31!important;border:1px solid #cad5c1;border-radius:12px}
[role="option"] {background:#fff!important;color:#203d31!important}
[role="option"] :is(div,span,p) {color:inherit!important}
[role="option"]:hover,[role="option"]:focus,[role="option"][aria-selected="true"] {
    background:#e5efdc!important;color:#193e2b!important;
}
[data-baseweb="popover"] li {background:#fff;color:#203d31}
[data-baseweb="popover"] li:hover {background:#e5efdc!important;color:#193e2b!important}
/* Sidebar sensitivity select: readable before click, on hover, and while open. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color:#ffffff!important;
    color:#203d31!important;
    border:1.5px solid #9cad91!important;
    border-radius:12px!important;
    min-height:46px;
    box-shadow:0 2px 6px #203d310b!important;
    cursor:pointer!important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    background-color:#edf5e8!important;
    border-color:#245b43!important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    background-color:#ffffff!important;
    border-color:#245b43!important;
    box-shadow:0 0 0 3px #245b4324!important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] :is(div,span,input,svg) {
    color:#203d31!important;
    -webkit-text-fill-color:#203d31!important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill:currentColor!important;
}
/* Dropdown menu is mounted outside the sidebar in some Streamlit versions. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [data-baseweb="menu"] {
    background:#ffffff!important;
    color:#203d31!important;
}
[data-baseweb="popover"] [role="option"] :is(div,span,p) {
    color:#203d31!important;
    -webkit-text-fill-color:#203d31!important;
}

[data-baseweb="radio"],[data-baseweb="checkbox"] {color:#203d31!important}
[data-baseweb="radio"]>div:first-of-type,[data-baseweb="checkbox"]>span:first-of-type {
    background-color:#fff!important;border-color:#819576!important;
}
[data-baseweb="radio"]:has(input:checked)>div:first-of-type,
[data-baseweb="checkbox"]:has(input:checked)>span:first-of-type {
    background-color:#245b43!important;border-color:#245b43!important;
}
[data-baseweb="radio"]:has(input:checked)>div:first-of-type>div {background-color:#fff!important}
[data-baseweb="checkbox"]:has(input:checked) svg {color:#fff!important}
[data-baseweb="radio"]:has(input:focus-visible),[data-baseweb="checkbox"]:has(input:focus-visible) {
    outline:2px solid #52734b;outline-offset:4px;border-radius:6px;
}
[role="slider"] {background:#245b43!important;border:2px solid #fff!important;color:#fff!important;box-shadow:0 0 0 1px #52734b}
[data-testid="stSliderTickBar"],[data-testid="stSliderThumbValue"] {color:#203d31!important}
[data-testid="stExpander"] summary {background:#fff!important;color:#203d31!important;border-radius:14px}
[data-testid="stExpander"] summary:hover {background:#eaf2e2!important;color:#193e2b!important}
[data-testid="stExpander"] summary :is(p,span,svg) {color:inherit!important}
[data-testid="stFileUploaderDropzone"] small {color:#586a5c!important}
[data-baseweb="tab"] {background:transparent!important;color:#405b40!important}
[data-baseweb="tab"]:hover {background:#dfe9d6!important;color:#193e2b!important}
[data-baseweb="tab"][aria-selected="true"] {background:#fff!important;color:#214a34!important}
[data-baseweb="tab"] :is(p,span,[data-testid="stMarkdownContainer"]) {color:inherit!important}
.topbar {display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:22px}
.logo {display:flex;align-items:center;gap:10px;font-weight:800;font-size:.9rem;letter-spacing:.05em}
.logo-icon {background:var(--green);color:var(--lime);border-radius:12px;width:37px;height:37px;display:grid;place-items:center;font-size:1.5rem}
.topnote {font-size:.68rem;color:var(--muted);letter-spacing:.08em}
.hero {display:grid;grid-template-columns:1.3fr 1fr;align-items:center;background:#133b2b;border-radius:28px;overflow:hidden;min-height:340px;margin-bottom:22px;box-shadow:0 12px 35px #153e2210}
.hero-copy {padding:40px 44px;position:relative;z-index:1}
.eyebrow {font-size:.64rem;letter-spacing:.18em;text-transform:uppercase;color:#bbd0a5;display:flex;align-items:center;gap:9px}
.dot {width:6px;height:6px;border-radius:50%;background:#d2e8a3;box-shadow:0 0 0 5px #d2e8a310;display:inline-block}
.hero h1 {color:#f2f4e5!important;font-family:Georgia,serif;font-size:clamp(2.8rem,4.1vw,4.1rem);font-weight:400;line-height:1.12;margin:22px 0 17px;letter-spacing:-.045em}
.hero p {color:#c4d3bf;font-size:.91rem;line-height:1.85;max-width:460px;margin:0}
.tags {display:flex;gap:8px;flex-wrap:wrap;margin-top:23px}.tags span {font-size:.65rem;padding:7px 12px;border:1px solid #acc39740;border-radius:50px;color:#dce7cb}
.art {position:relative;padding:25px 25px 20px;background:radial-gradient(ellipse at center,#54704460,transparent 70%)}
.art svg {width:100%;height:270px;filter:drop-shadow(0 20px 20px #001b1740)}
.art-caption {display:flex;justify-content:space-between;color:#b7c99c;font-family:monospace;font-size:.57rem;letter-spacing:.1em;padding:0 30px}
.stats {display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 26px}.stat {background:#fff;border:1px solid var(--line);border-radius:16px;padding:17px 21px;display:flex;align-items:center;gap:14px}.stat-icon {background:#edf3e5;color:#608245;width:39px;height:39px;display:grid;place-items:center;border-radius:12px;font-size:1.2rem}.stat strong {display:block;font-size:1.1rem;font-weight:650}.stat small {font-size:.69rem;color:var(--muted)}
[data-baseweb="tab-list"] {background:#e9eee2;border:1px solid #dde5d5;border-radius:14px;padding:5px;gap:5px;width:fit-content;margin-bottom:20px}
[data-baseweb="tab"] {padding:10px 24px;height:auto;border-radius:10px;color:#60715e}
[data-baseweb="tab"][aria-selected="true"] {background:#fff;color:#214a34;box-shadow:0 2px 7px #193c1510}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none}
.section-head {display:flex;align-items:center;gap:12px;margin:6px 0 18px}.step {border:1px solid #cedac1;color:#617c4d;border-radius:9px;padding:5px 8px;font-size:.67rem;font-family:monospace}.section-head h3 {margin:0;font-size:1.12rem}.section-head small {display:block;font-size:.72rem;color:var(--muted);margin-top:5px}
.card {background:#fff;border:1px solid var(--line);border-radius:21px;padding:25px;margin:0 0 16px;animation:enter .3s ease-out}
.kicker {font-size:.64rem;letter-spacing:.13em;color:#76876c;text-transform:uppercase;margin-bottom:15px}
.empty {min-height:310px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;background:radial-gradient(circle at 50% 30%,#eaf1de,transparent 70%),#fff;border:1px solid var(--line);border-radius:21px;padding:30px}
.orbit {position:relative;display:grid;place-items:center;width:116px;height:116px;border:1px dashed #b1c397;border-radius:50%;margin-bottom:22px}.orbit:before {content:'';position:absolute;inset:13px;border:1px solid #d6e1c6;border-radius:50%}.orbit span {font-size:2rem;color:#6d8a50}.empty h3 {margin:0 0 10px}.empty p {font-size:.83rem;color:var(--muted);max-width:300px;line-height:1.8;margin:0}
.helper {background:#edf2e7;border-radius:14px;padding:17px 19px;font-size:.78rem;color:#637458;line-height:1.8;margin-top:15px}
.badge {display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:50px;background:var(--tint);color:var(--accent);font-size:.76rem;font-weight:700}
.result-grid {display:flex;justify-content:space-between;align-items:center;gap:18px;margin:20px 0}.result-grid h2 {font-size:1.8rem;margin:0 0 8px;overflow-wrap:anywhere}.result-grid p {font-size:.84rem;color:var(--muted);margin:0;line-height:1.7}
.ring {flex-shrink:0;width:120px;height:120px;display:grid;place-items:center;border-radius:50%;background:conic-gradient(var(--accent) var(--angle),#edf0e7 0);direction:ltr;box-shadow:0 0 0 7px #f8faf4}.ring-inner {width:101px;height:101px;border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center}.ring strong {font-size:1.9rem;font-weight:500}.ring small {font-size:.63rem;color:var(--muted)}
.advice {border-top:1px solid var(--line);padding-top:16px;font-size:.82rem;line-height:1.8;color:#6b7966}
.prediction {display:flex;justify-content:space-between;gap:12px;font-size:.79rem}.prediction span {overflow-wrap:anywhere}.track {background:#edf1e7;height:7px;border-radius:20px;margin:10px 0 20px;overflow:hidden}.fill {height:100%;border-radius:20px;animation:fill .6s ease-out}
.scan-image {background:#edf1e6;border:1px solid #dce4d1;border-radius:20px;overflow:hidden;position:relative;margin:14px 0}.scan-image img {display:block;width:100%;max-height:345px;object-fit:contain}.image-meta {display:flex;justify-content:space-between;font-family:monospace;font-size:.62rem;padding:12px 16px;color:#697c5c;border-top:1px solid #dce4d1}
.journal {display:flex;justify-content:space-between;align-items:center;gap:16px}.journal h3 {margin:9px 0 5px}.journal small {color:var(--muted);font-size:.74rem}.journal-value {font-size:1.2rem;font-weight:650;color:#476b37;white-space:nowrap}
.footer {display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line);padding-top:20px;margin-top:30px;font-size:.67rem;color:#7d8b73}
[data-testid="stMetric"] {border:1px solid var(--line);border-radius:16px;background:#fff;padding:17px;color:var(--ink)}
@keyframes enter {from {opacity:0;transform:translateY(6px)}to {opacity:1;transform:translateY(0)}}
@keyframes fill {from {max-width:0}to {max-width:100%}}
@media(prefers-reduced-motion:reduce) {* {animation:none!important;transition:none!important}}
@media(max-width:800px) {.hero-copy {padding:30px}.hero h1 {font-size:2.9rem}.art {padding:10px}.stats {gap:8px}.stat {padding:14px;gap:9px}.topnote {display:none}}
@media(max-width:600px) {.block-container {padding:1.3rem 1rem}.hero {grid-template-columns:1fr}.hero-copy {padding:30px 25px}.art {display:none}.hero h1 {font-size:2.7rem}.stats {gap:6px}.stat {padding:12px 9px;display:block}.stat-icon {display:none}.stat strong {font-size:.95rem}.stat small {font-size:.6rem}[data-baseweb="tab"] {padding:8px 12px}.card {padding:20px}.ring {width:99px;height:99px}.ring-inner {width:83px;height:83px}.result-grid h2 {font-size:1.4rem}.footer {flex-direction:column}}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)
if lang == 'ar':
    st.markdown('<style>.block-container,[data-testid="stSidebar"] {direction:rtl;text-align:right}.hero h1 {font-family:Tahoma,Arial,sans-serif;font-size:clamp(2.3rem,3.6vw,3.4rem);font-weight:700;line-height:1.48;letter-spacing:-.04em}.eyebrow,.kicker,.logo,.topnote {letter-spacing:0}</style>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### ◈ CROP DOCTOR')
    st.caption(tr('A thoughtful space for healthier plants.', 'مساحة ذكية للعناية بصحة نباتاتك.'))
    st.radio('Language / اللغة',['ar','en'],format_func=lambda x:'العربية' if x=='ar' else 'English',key='lang',horizontal=True)
    st.divider()
    st.markdown('#### '+tr('Analysis preferences','تفضيلات التحليل'))
    preset=st.selectbox(tr('Sensitivity','حساسية التحليل'),['balanced','careful','custom'],format_func=lambda x:{'balanced':tr('Balanced','متوازن'),'careful':tr('More cautious','أكثر تحفظًا'),'custom':tr('Custom','تخصيص')}[x])
    if preset == 'custom':
        conf_t=st.slider(t['conf'],.50,.99,.80,.01)
        ent_t=st.slider(t['ent'],.30,2.,1.,.05)
    else:
        conf_t,ent_t=(.80,1.) if preset=='balanced' else (.90,.70)
        st.caption(tr(f'Confidence ≥ {conf_t:.0%} · Entropy ≤ {ent_t:.2f}',f'الثقة ≥ {conf_t:.0%} · التشتت ≤ {ent_t:.2f}'))
    st.caption(tr('Current results update with these preferences. Saved scans retain their original settings.','تتغير النتيجة الحالية مع هذه التفضيلات. يحتفظ السجل بإعدادات كل فحص وقت إجرائه.'))
    with st.expander(tr('A better photo, a better start','صورة أفضل، بداية أفضل')):
        for tip in t['tips_l']:
            st.write('✓ '+tip)
    st.divider()
    st.caption(tr('Built by Yahya Majrashi','تطوير يحيى الماجرشي'))

LEAF='''<svg viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg" fill="none" aria-hidden="true"><defs><linearGradient id="lg" x1="100" y1="280" x2="315" y2="40" gradientUnits="userSpaceOnUse"><stop stop-color="#356849"/><stop offset="1" stop-color="#d9eeb4"/></linearGradient></defs><g stroke="#bdd29e" stroke-opacity=".2"><circle cx="210" cy="160" r="132"/><circle cx="210" cy="160" r="150" stroke-dasharray="2 9"/><path d="M20 160H400M210 12V310"/></g><path d="M123 265C64 161 144 56 313 39C339 190 256 282 123 265Z" fill="url(#lg)"/><g stroke="#e6f2c8" stroke-opacity=".6" stroke-width="1.6" stroke-linecap="round"><path d="M102 294L290 70M143 246L136 166M175 206L168 119M207 169L205 87M240 128L244 64M143 246L229 243M175 206L272 195M207 169L301 146M240 128L313 105"/></g><circle cx="210" cy="160" r="6" fill="#e0efbf"/><circle cx="210" cy="160" r="17" stroke="#e0efbf" stroke-opacity=".7"/><path d="M62 98V70H90M333 252H361V224" stroke="#c7db9c" stroke-width="2"/></svg>'''
st.markdown(f'<div class="topbar"><div class="logo"><span class="logo-icon">❧</span>CROP DOCTOR</div><div class="topnote">{tr("BOTANICAL INTELLIGENCE / PERSONAL PLANT STUDIO","ذكاء نباتي / مساحتك للعناية بالنبات")}</div></div>',unsafe_allow_html=True)
st.markdown(f'<div class="hero"><div class="hero-copy"><div class="eyebrow"><span class="dot"></span>{tr("SMALL SIGNS. MEANINGFUL INSIGHTS.","تفاصيل صغيرة. معرفة تصنع فرقًا.")}</div><h1>{tr("Healthy starts<br>with a closer look.","صحة نباتك تبدأ<br>بنظرة أقرب.")}</h1><p>{tr("Every leaf has something to tell you. Explore plant health with clear, confidence-aware AI insights.","لكل ورقة ما تقوله. اكتشف حالة نباتك بتحليل ذكي، ونتائج واضحة تعرف متى تكون غير متأكدة.")}</p><div class="tags"><span>✦ {tr("Leaf analysis","تحليل الأوراق")}</span><span>◉ {tr("38 model classes","38 فئة في النموذج")}</span></div></div><div class="art">{LEAF}<div class="art-caption"><span>FOLIUM / 001</span><span>OBSERVE · UNDERSTAND</span></div></div></div>',unsafe_allow_html=True)
summary=st.empty()

def draw_summary():
    n=len(st.session_state.history)
    healthy=sum(x['status']=='healthy' for x in st.session_state.history)
    uncertain=sum(x['status']=='uncertain' for x in st.session_state.history)
    values=[('⌁',n,tr('Scans this session','فحوصات الجلسة')),('❧',healthy,tr('Healthy results','نتائج سليمة')),('◎',uncertain,tr('Uncertain results','نتائج غير مؤكدة'))]
    summary.markdown('<div class="stats">'+''.join(f'<div class="stat"><span class="stat-icon">{icon}</span><div><strong>{value:02d}</strong><small>{label}</small></div></div>' for icon,value,label in values)+'</div>',unsafe_allow_html=True)

def heading(number,title,sub):
    st.markdown(f'<div class="section-head"><span class="step">{number}</span><div><h3>{title}</h3><small>{sub}</small></div></div>',unsafe_allow_html=True)

def classify(p,labels):
    order=p.argsort()[::-1][:3]
    conf=float(p[order[0]])
    entropy=float(-(p*np.log(p+1e-9)).sum())
    crop,cond=pretty(labels[int(order[0])])
    status='uncertain' if conf<conf_t or entropy>ent_t else 'healthy' if cond.lower()=='healthy' else 'disease'
    return order,conf,entropy,crop,cond,status

scan_tab,journal_tab,about_tab=st.tabs([tr('◉  Plant scan','◉  فحص النبات'),tr('▤  My journal','▤  سجل الفحوصات'),tr('✧  Discover','✧  عن التجربة')])
with scan_tab:
    left,right=st.columns([1,1.06],gap='large')
    img=None
    signature=None
    with left:
        heading('01',tr('Meet your leaf','تعرّف على ورقتك'),tr('A clear photo is the best starting point.','صورة واضحة هي أفضل نقطة بداية.'))
        mode=st.radio(tr('Image source','مصدر الصورة'),['upload','camera'],format_func=lambda x:tr('Upload photo','رفع صورة') if x=='upload' else tr('Use camera','استخدام الكاميرا'),horizontal=True,label_visibility='collapsed')
        file=st.file_uploader(tr('Drop your leaf photo here','ضع صورة الورقة هنا'),type=['jpg','jpeg','png'],key=f'upload_{st.session_state.upload_key}') if mode=='upload' else st.camera_input(tr('Photograph one leaf','صوّر ورقة واحدة'),key=f'camera_{st.session_state.upload_key}')
        if file:
            try:
                raw=file.getvalue()
                img=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
                rotation=st.select_slider(tr('Rotate photo','تدوير الصورة'),options=[0,90,180,270],value=0,format_func=lambda x:f'{x}°',key=f'rotation_{hashlib.sha256(raw).hexdigest()[:12]}')
                if rotation:
                    img=img.rotate(-rotation,expand=True)
                signature=hashlib.sha256(raw+str(rotation).encode()).hexdigest()
                preview=img.copy();preview.thumbnail((900,700))
                buf=io.BytesIO();preview.save(buf,format='JPEG',quality=88)
                b64=base64.b64encode(buf.getvalue()).decode()
                st.markdown(f'<div class="scan-image"><img alt="{tr("Uploaded leaf preview","معاينة الورقة المرفوعة")}" src="data:image/jpeg;base64,{b64}"><div class="image-meta"><span>{img.width} × {img.height} PX</span><span>{len(raw)/1024:.0f} KB · {rotation}°</span></div></div>',unsafe_allow_html=True)
            except Exception:
                img=None
                st.error(tr('Unable to read this image. Try another JPG or PNG.','تعذر قراءة الصورة. جرّب صورة JPG أو PNG أخرى.'))
        if img is None:
            st.markdown(f'<div class="helper">☀ &nbsp; {tr("Natural light. One leaf. A simple background. Small details make your photo more useful.","إضاءة طبيعية، ورقة واحدة، وخلفية بسيطة. تفاصيل صغيرة تجعل صورتك أكثر فائدة.")}</div>',unsafe_allow_html=True)
        analyze=st.button(tr('Analyze this leaf  →','تحليل هذه الورقة  ←'),type='primary',use_container_width=True,disabled=img is None)
        if analyze:
            try:
                with st.spinner(tr('Reading your leaf…','جارٍ قراءة تفاصيل الورقة…')):
                    scores,ms,labels=predict(img)
                st.session_state.result={'signature':signature,'scores':scores,'ms':ms,'labels':labels}
                _,conf,entropy,crop,cond,state=classify(scores,labels)
                st.session_state.history.append({'time':datetime.now().isoformat(timespec='seconds'),'crop':crop,'condition':cond,'confidence':round(conf,4),'entropy':round(entropy,4),'status':state,'confidence_threshold':conf_t,'entropy_threshold':ent_t})
            except Exception as exc:
                st.session_state.result=None
                st.error(tr('Analysis could not run. Check TensorFlow, model.tflite and labels.json.','تعذر تشغيل التحليل. تأكد من تثبيت TensorFlow ووجود model.tflite و labels.json بجانب التطبيق.'))
                with st.expander(tr('Error details','تفاصيل الخطأ')):
                    st.code(str(exc))
        if file and st.button(tr('Start another scan','بدء فحص آخر'),use_container_width=True):
            st.session_state.upload_key+=1
            st.session_state.result=None
            st.rerun()
    with right:
        heading('02',tr('A clearer picture','صورة أوضح لصحة النبات'),tr('Your results, explained simply.','نتائجك، بتفسير بسيط وواضح.'))
        result=st.session_state.result
        if img is None or result is None or result['signature']!=signature:
            st.markdown(f'<div class="empty"><div class="orbit"><span>❧</span></div><h3>{tr("Ready when you are","جاهز عندما تكون جاهزًا")}</h3><p>{tr("Add a leaf photo, then start your scan. Your plant insights will appear right here.","أضف صورة ورقة وابدأ الفحص. ستظهر هنا حالة النبات ونسبة الثقة وأقرب الاحتمالات.")}</p></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="helper">✧ &nbsp; {tr("Uncertainty is useful information. The app holds back when a prediction is not clear enough.","عدم اليقين معلومة مهمة. يمتنع التطبيق عن الجزم عندما لا تكون النتيجة واضحة بما يكفي.")}</div>',unsafe_allow_html=True)
        else:
            order,conf,entropy,crop,cond,state=classify(result['scores'],result['labels'])
            color,tint,icon={'healthy':('#39784d','#e8f3e5','✓'),'disease':('#b85939','#f9ebe1','!'),'uncertain':('#947322','#f8f1d9','?')}[state]
            title=t['uncertain'] if state=='uncertain' else crop
            subtitle=tr('Try a clearer photo to reduce uncertainty.','جرّب صورة أوضح لتقليل عدم اليقين.') if state=='uncertain' else cond
            advice=t['adv_u'] if state=='uncertain' else t['adv_d'] if state=='disease' else tr('Keep observing your plant for changes.','استمر في مراقبة النبات لأي تغيّرات.')
            st.markdown(f'<div class="card" style="--accent:{color};--tint:{tint};--angle:{conf*360:.2f}deg"><div class="kicker">{tr("YOUR PLANT REPORT","تقرير حالة النبات")}</div><span class="badge">{icon} {t[state]}</span><div class="result-grid"><div><h2>{e(title)}</h2><p>{e(subtitle)}</p></div><div class="ring"><div class="ring-inner"><strong>{conf:.0%}</strong><small>{t["confl"]}</small></div></div><div class="advice">{advice}</div></div>',unsafe_allow_html=True)
            rows=''
            for rank,i in enumerate(order):
                value=float(result['scores'][i])
                label=' · '.join(pretty(result['labels'][int(i)]))
                rows+=f'<div class="prediction"><span>{rank+1:02d} &nbsp; {e(label)}</span><b>{value:.1%}</b></div><div class="track"><div class="fill" style="width:{value*100:.2f}%;background:{color if rank==0 else "#afc29b"}"></div></div>'
            st.markdown(f'<div class="card"><div class="kicker">{t["top3"]}</div>{rows}</div>',unsafe_allow_html=True)
            with st.expander(tr('Understand this result','افهم هذه النتيجة')):
                st.write(tr('Confidence is the model’s score, not a guarantee of correctness.','الثقة درجة يعطيها النموذج وليست ضمانًا لصحة النتيجة.'))
                st.write(f'{t["confl"]}: {conf:.1%} / {conf_t:.0%}')
                st.write(f'Entropy: {entropy:.3f} / {ent_t:.2f}')
                st.caption(f'{t["time"]}: {result["ms"]:.0f} ms')
                if conf<conf_t: st.warning(t['why_c'])
                if entropy>ent_t: st.warning(t['why_e'])
            report={'crop':crop,'condition':cond,'confidence':conf,'entropy':entropy,'status':state,'confidence_threshold':conf_t,'entropy_threshold':ent_t}
            st.download_button(tr('↓  Save this report','↓  حفظ هذا التقرير'),json.dumps(report,ensure_ascii=False,indent=2),'crop_doctor_report.json','application/json',use_container_width=True)

with journal_tab:
    heading('03',tr('Your plant journal','سجل نباتاتك'),tr('Every scan, in one calm space.','كل فحوصاتك في مساحة واحدة.'))
    st.caption(tr('History lasts for this session. Export it to keep a copy. Repeating analysis creates a new entry.','السجل خاص بالجلسة الحالية. صدّره للاحتفاظ بنسخة. إعادة التحليل تضيف فحصًا جديدًا.'))
    if not st.session_state.history:
        st.markdown(f'<div class="empty"><div class="orbit"><span>▤</span></div><h3>{tr("Your story starts with a scan","تبدأ الحكاية بفحص")}</h3><p>{tr("Completed scans will appear here, ready to revisit or export.","ستظهر الفحوصات المكتملة هنا، لتراجعها أو تحفظها لاحقًا.")}</p></div>',unsafe_allow_html=True)
    else:
        df=pd.DataFrame(st.session_state.history)
        a,b=st.columns([2,1])
        query=a.text_input(tr('Search by crop or condition','ابحث بالمحصول أو الحالة'),placeholder=tr('Try tomato…','جرّب Tomato…'))
        status=b.selectbox(tr('Show results','عرض النتائج'),['all','healthy','disease','uncertain'],format_func=lambda x:tr('All results','كل النتائج') if x=='all' else t[x])
        filtered=df.copy()
        if query:
            filtered=filtered[filtered.crop.str.contains(query,case=False,regex=False)|filtered.condition.str.contains(query,case=False,regex=False)]
        if status!='all':filtered=filtered[filtered.status==status]
        view=st.radio(tr('Display','طريقة العرض'),['cards','table'],format_func=lambda x:tr('Cards','بطاقات') if x=='cards' else tr('Table','جدول'),horizontal=True)
        if filtered.empty:st.info(tr('No matching scans.','لا توجد فحوصات مطابقة.'))
        elif view=='table':st.dataframe(filtered.iloc[::-1],hide_index=True,use_container_width=True)
        else:
            for _,row in filtered.iloc[::-1].head(30).iterrows():
                st.markdown(f'<div class="card journal"><div><div class="kicker">{e(row["time"])} · {t[row["status"]]}</div><h3>{e(row["crop"])}</h3><small>{e(row["condition"])}</small></div><div class="journal-value">{row["confidence"]:.0%}</div></div>',unsafe_allow_html=True)
            if len(filtered)>30:st.caption(tr('Showing the latest 30. Use table view for all entries.','تُعرض أحدث 30 نتيجة. اختر الجدول لعرض جميع النتائج.'))
        st.download_button(tr('↓  Export filtered history','↓  تصدير السجل المُصفّى'),filtered.to_csv(index=False).encode('utf-8-sig'),'crop_doctor_history.csv','text/csv')
        with st.expander(tr('Manage history','إدارة السجل')):
            confirm=st.checkbox(tr('I want to clear this session’s history','أريد مسح سجل هذه الجلسة'))
            if st.button(t['clr'],disabled=not confirm):
                st.session_state.history=[]
                st.rerun()

with about_tab:
    heading('✧',tr('Thoughtful technology. Clear boundaries.','تقنية ذكية. وحدود واضحة.'),tr('Get to know the intelligence behind your scan.','تعرّف على التقنية وراء فحصك.'))
    c1,c2=st.columns(2,gap='large')
    with c1:
        st.markdown(f'<div class="card"><div class="kicker">{tr("THE PROCESS","كيف تعمل التجربة")}</div><h3>{tr("From photo to insight","من الصورة إلى المعرفة")}</h3><p>{t["how_t"]}</p></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="kicker">{tr("GOOD TO KNOW","من المهم معرفته")}</div><h3>{t["lim"]}</h3>'+''.join(f'<p>• {e(x)}</p>' for x in t['lim_t'])+'</div>',unsafe_allow_html=True)
    with st.expander(tr('Original project metrics','مقاييس المشروع الأصلي')):
        st.caption(tr('Values retained from the original app; not independently verified.','الأرقام من التطبيق الأصلي؛ لم تُتحقق بشكل مستقل.'))
        for col,key,val in zip(st.columns(4),['m1','m2','m3','m4'],['95.9%','2.58 MB','92.0%','98.7%']):col.metric(t[key],val)
    st.info(tr('Local installation can run offline once dependencies are installed. Online deployments process images on the hosting server.','يعمل التثبيت المحلي دون إنترنت بعد تثبيت المتطلبات. عند النشر على الإنترنت، تُعالج الصور على خادم الاستضافة.'))
draw_summary()
st.markdown(f'<div class="footer"><span>❧ &nbsp; CROP DOCTOR · {tr("Care begins with understanding.","العناية تبدأ بالفهم.")}</span><span>{t["by"]}</span></div>',unsafe_allow_html=True)
