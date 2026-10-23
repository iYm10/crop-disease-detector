import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE = Path(__file__).parent
IMG = 224

st.set_page_config(page_title="Crop Doctor", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------- text (EN / AR)
T = {
    "en": {
        "title": "Crop Doctor", "sub": "Offline AI that spots plant diseases from a single leaf photo, and tells you when it is not sure.",
        "lang": "Language", "settings": "Settings", "conf": "Minimum confidence", "conf_h": "Below this, the app says 'uncertain' instead of guessing.",
        "ent": "Maximum uncertainty (entropy)", "ent_h": "Rejects confusing or non-leaf images.",
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
        "ent": "أقصى درجة تشتت (entropy)", "ent_h": "يرفض الصور المربكة أو التي ليست أوراقًا.",
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

# ---------------------------------------------------------------- model
@st.cache_resource
def load():
    it = tf.lite.Interpreter(model_path=str(BASE / "model.tflite"))
    it.allocate_tensors()
    with open(BASE / "labels.json") as f:
        return it, json.load(f), threading.Lock()

interp, labels, lock = load()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

def predict(img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG, IMG))
    x = np.asarray(img, dtype="float32")[None]
    t0 = time.perf_counter()
    with lock:
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        p = interp.get_tensor(out["index"])[0].copy()
    return p, (time.perf_counter() - t0) * 1000

def pretty(name):
    crop, cond = name.split("___")
    return crop.replace("_", " ").strip(), cond.replace("_", " ").strip()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    lang = "ar" if st.radio("Language / اللغة", ["English", "العربية"], horizontal=True) == "العربية" else "en"
    t = T[lang]
    st.subheader(f"⚙️ {t['settings']}")
    conf_t = st.slider(t["conf"], 0.50, 0.99, 0.80, 0.01, help=t["conf_h"])
    ent_t = st.slider(t["ent"], 0.30, 2.00, 1.00, 0.05, help=t["ent_h"])
    st.divider()
    st.markdown(f"**{t['tips']}**")
    for tip in t["tips_l"]:
        st.markdown(f"- {tip}")

rtl = "direction:rtl;text-align:right;" if lang == "ar" else ""
st.markdown(f"""
<style>
.block-container {{padding-top:2rem; {rtl}}}
.hero {{background:linear-gradient(135deg,#14532d 0%,#16a34a 60%,#84cc16 100%);color:#fff;padding:28px 32px;border-radius:20px;margin-bottom:18px}}
.hero h1 {{margin:0;font-size:2.2rem;color:#fff}}
.hero p {{margin:6px 0 0;font-size:1.05rem;opacity:.95;color:#fff}}
.card {{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:18px 20px;background:rgba(128,128,128,.07)}}
.badge {{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:700;font-size:.9rem;color:#fff}}
.b-ok {{background:#16a34a}} .b-bad {{background:#dc2626}} .b-unk {{background:#d97706}}
.big {{font-size:1.7rem;font-weight:800;margin:10px 0 2px}}
.muted {{opacity:.75;font-size:.92rem}}
.track {{background:rgba(128,128,128,.25);border-radius:999px;height:12px;overflow:hidden;margin:4px 0 12px}}
.fill {{height:12px;border-radius:999px}}
.row {{display:flex;justify-content:space-between;font-size:.93rem}}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="hero"><h1>🌿 {t["title"]}</h1><p>{t["sub"]}</p></div>', unsafe_allow_html=True)

def bar(label, value, color):
    return (f'<div class="row"><span>{label}</span><b>{value:.1%}</b></div>'
            f'<div class="track"><div class="fill" style="width:{value*100:.1f}%;background:{color}"></div></div>')

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.seen = set()

tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

# ---------------------------------------------------------------- diagnose
with tab1:
    mode = st.radio(t["src"], [t["up"], t["cam"]], horizontal=True)
    file = st.file_uploader(t["up_l"], type=["jpg", "jpeg", "png"]) if mode == t["up"] else st.camera_input(t["cam_l"])

    if not file:
        st.info(t["empty"])
    else:
        raw = file.getvalue()
        img = Image.open(file)
        p, ms = predict(img)
        order = p.argsort()[::-1][:3]
        conf = float(p[order[0]])
        ent = float(-(p * np.log(p + 1e-9)).sum())
        crop, cond = pretty(labels[order[0]])

        low_conf, high_ent = conf < conf_t, ent > ent_t
        if low_conf or high_ent:
            state, badge, color = "uncertain", "b-unk", "#d97706"
        elif cond.lower() == "healthy":
            state, badge, color = "healthy", "b-ok", "#16a34a"
        else:
            state, badge, color = "disease", "b-bad", "#dc2626"

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.image(img)
        with right:
            if state == "uncertain":
                why = t["why_c"] if low_conf else t["why_e"]
                st.markdown(f'<div class="card"><span class="badge {badge}">⚠️ {t["uncertain"]}</span>'
                            f'<div class="big">{conf:.0%}</div><div class="muted">{why}</div>'
                            f'<p style="margin-top:12px">{t["adv_u"]}</p></div>', unsafe_allow_html=True)
            else:
                icon = "✅" if state == "healthy" else "🦠"
                advice = "" if state == "healthy" else f'<p style="margin-top:12px">{t["adv_d"]}</p>'
                st.markdown(f'<div class="card"><span class="badge {badge}">{icon} {t[state]}</span>'
                            f'<div class="big">{crop}</div><div class="muted">{cond}</div>'
                            f'<div style="margin-top:14px">{bar(t["confl"], conf, color)}</div>{advice}</div>',
                            unsafe_allow_html=True)

            st.markdown(f"##### {t['top3']}")
            rows = "".join(bar(" - ".join(pretty(labels[i])), float(p[i]), color if k == 0 else "#94a3b8")
                           for k, i in enumerate(order))
            st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

            with st.expander(t["details"]):
                c1, c2, c3 = st.columns(3)
                c1.metric(t["confl"], f"{conf:.1%}")
                c2.metric("Entropy", f"{ent:.2f}")
                c3.metric(t["time"], f"{ms:.0f} ms")

        key = hashlib.md5(raw).hexdigest()
        if key not in st.session_state.seen:
            st.session_state.seen.add(key)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"), "crop": crop, "condition": cond,
                "confidence": round(conf, 3), "status": state})

# ---------------------------------------------------------------- history
with tab2:
    if not st.session_state.history:
        st.info(t["hist_empty"])
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(t["dl"], df.to_csv(index=False).encode("utf-8"), "crop_doctor_history.csv", "text/csv")
        if c2.button(t["clr"]):
            st.session_state.history, st.session_state.seen = [], set()
            st.rerun()

# ---------------------------------------------------------------- about
with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["m1"], "95.9%")
    m2.metric(t["m2"], "2.58 MB")
    m3.metric(t["m3"], "92.0%")
    m4.metric(t["m4"], "98.7%")
    st.markdown(f"### {t['how']}")
    st.write(t["how_t"])
    st.markdown(f"### {t['lim']}")
    for item in t["lim_t"]:
        st.markdown(f"- {item}")
    st.divider()
    st.caption(f"{t['by']} · MobileNetV2 · TensorFlow Lite · PlantVillage")
