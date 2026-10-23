import html
import io
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

# ---------------------------------------------------------------- model
@st.cache_resource
def load():
    it = tf.lite.Interpreter(model_path=str(BASE / "model.tflite"))
    it.allocate_tensors()
    with open(BASE / "labels.json") as f:
        return it, json.load(f), threading.Lock()

try:
    interp, labels, lock = load()
except Exception as exc:
    st.error("تعذر تحميل النموذج. ضع model.tflite و labels.json بجانب ملف التطبيق. / Model files could not be loaded.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()
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

# ---------------------------------------------------------------- interface

def tr(en, ar):
    return ar if lang == "ar" else en


def esc(value):
    return html.escape(str(value))


for key, default in {"history": [], "seen": set(), "prediction": None, "upload_id": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 🌿 CROP DOCTOR")
    lang = st.radio("Language / اللغة", ["en", "ar"], format_func=lambda x: "العربية" if x == "ar" else "English", horizontal=True, key="language")
    t = T[lang]
    st.caption(tr("YOUR PLANT HEALTH COMPANION", "رفيقك لصحة النبات"))
    st.divider()
    st.subheader(t["settings"])
    conf_t = st.slider(t["conf"], .50, .99, .80, .01, help=t["conf_h"])
    ent_t = st.slider(t["ent"], .30, 2., 1., .05, help=t["ent_h"])
    st.caption(tr("Adjust thresholds to see the current result update instantly.", "حرّك المؤشرات وشاهد كيف تتغير حالة النتيجة مباشرةً."))
    st.divider()
    st.markdown(f"**{t['tips']}**")
    for tip in t["tips_l"]:
        st.markdown(f"✓ {tip}")
    st.divider()
    st.caption(t["by"])

st.markdown('''<style>
.stApp {background:var(--background-color)}
.block-container {max-width:1240px;padding-top:2rem;padding-bottom:2rem}
h1,h2,h3 {letter-spacing:-.035em}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#102e29,#1c5245);padding:44px;border-radius:26px;color:#fff;margin:12px 0 24px}
.hero:after {content:'❧';position:absolute;right:3%;top:-65px;font-size:320px;line-height:1.3;color:#c7dfab;opacity:.10;pointer-events:none}
.hero h1 {font-size:clamp(2.2rem,5vw,3.7rem);line-height:1.1;color:#fff!important;margin:16px 0;max-width:800px;position:relative;z-index:1}
.hero p {color:#d3e5da;max-width:640px;line-height:1.8;margin:0;position:relative;z-index:1}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#c3dea1}
.pills {display:flex;gap:9px;flex-wrap:wrap;margin-top:24px}
.pill {font-size:.76rem;border:1px solid #ffffff30;padding:7px 13px;border-radius:30px;color:#edf4e9;background:#ffffff08}
.card {border:1px solid #829a842e;background:var(--secondary-background-color);border-radius:20px;padding:25px;margin-bottom:16px;animation:arrive .35s ease-out}
.kicker {font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;opacity:.65;margin-bottom:10px}
.empty {text-align:center;padding:55px 24px;border:1px dashed #829a8466;border-radius:20px;margin-top:16px;background:var(--secondary-background-color)}
.empty-icon {font-size:3.7rem;margin-bottom:12px}
.empty h3 {margin:0 0 10px}.empty p {opacity:.65;font-size:.95rem}
.badge {display:inline-block;border-radius:30px;padding:7px 13px;font-size:.8rem;font-weight:700;background:var(--tint);color:var(--ink)}
.result-main {display:flex;align-items:center;justify-content:space-between;gap:18px;margin-top:23px}
.result-main h2 {margin:0 0 8px;font-size:1.9rem;overflow-wrap:anywhere}.result-main p {opacity:.7;margin:0}
.ring {flex-shrink:0;width:112px;height:112px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--ink) var(--angle),#8a9c8830 0);direction:ltr}
.ring-center {width:92px;height:92px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--secondary-background-color)}
.ring strong {font-size:1.6rem}.ring small {font-size:.65rem;opacity:.7}
.track {height:8px;border-radius:99px;background:#80918022;overflow:hidden;margin:9px 0 20px}.fill {height:100%;border-radius:99px;animation:grow .55s ease-out}
.pred-row {display:flex;justify-content:space-between;gap:12px;font-size:.88rem}.pred-row span {overflow-wrap:anywhere}
.note {font-size:.9rem;line-height:1.7;margin-top:20px;padding-top:16px;border-top:1px solid #829a8430}
.footer {text-align:center;opacity:.55;font-size:.75rem;margin-top:30px}
[data-testid="stFileUploader"] {border-radius:18px;overflow:hidden}
[data-testid="stImage"] img {border-radius:18px;max-height:430px;object-fit:contain}
[data-testid="stMetric"] {padding:18px;border:1px solid #829a8433;border-radius:16px;background:var(--secondary-background-color)}
.stButton button,.stDownloadButton button {border-radius:11px;transition:transform .18s}.stButton button:hover,.stDownloadButton button:hover {transform:translateY(-2px)}
button[kind="primary"] {background:#257357;color:white;border-color:#257357}
[data-baseweb="tab-list"] {gap:24px;margin-bottom:16px}
@keyframes arrive {from {opacity:0;transform:translateY(8px)}to {opacity:1;transform:translateY(0)}}
@keyframes grow {from {max-width:0}to {max-width:100%}}
@media(prefers-reduced-motion:reduce) {* {animation:none!important;transition:none!important}}
@media(max-width:640px) {.hero {padding:28px 22px}.card {padding:20px}.ring {width:94px;height:94px}.ring-center {width:76px;height:76px}.result-main h2 {font-size:1.4rem}}
</style>''', unsafe_allow_html=True)
if lang == "ar":
    st.markdown('''<style>.block-container,[data-testid="stSidebar"] {direction:rtl;text-align:right}.hero:after {right:auto;left:3%}.eyebrow,.kicker {letter-spacing:0}.pred-row {direction:rtl}</style>''', unsafe_allow_html=True)

st.markdown(f'''<div class="hero"><div class="eyebrow">{tr("PLANT HEALTH · POWERED BY AI", "صحة النبات · بدعم الذكاء الاصطناعي")}</div><h1>{t['title']}</h1><p>{t['sub']}</p><div class="pills"><span class="pill">{tr("One leaf. A clearer picture.", "ورقة واحدة. صورة أوضح.")}</span><span class="pill">MobileNetV2</span><span class="pill">{tr("38 plant conditions", "38 فئة نباتية")}</span></div></div>''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])
with tab1:
    left, right = st.columns([1, 1.12], gap="large")
    with left:
        st.markdown(f"### {tr('01 · Add your leaf', '01 · أضف صورة الورقة')}")
        mode = st.radio(t['src'], ['upload', 'camera'], format_func=lambda x: t['up'] if x == 'upload' else t['cam'], horizontal=True, key='source')
        file = st.file_uploader(t['up_l'], type=['jpg','jpeg','png'], key=f"upload_{st.session_state.upload_id}") if mode == 'upload' else st.camera_input(t['cam_l'], key=f"camera_{st.session_state.upload_id}")
        img, raw = None, None
        if file is not None:
            raw = file.getvalue()
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
                st.image(img, use_container_width=True)
                st.caption(f"{img.width} × {img.height} px · {len(raw)/1024:.0f} KB")
            except Exception:
                st.error(tr('This image cannot be opened. Try another JPG or PNG.', 'تعذر فتح الصورة. جرّب صورة JPG أو PNG أخرى.'))
        if img is None:
            st.markdown(f'''<div class="empty"><div class="empty-icon">🌱</div><h3>{tr('Start with a single leaf', 'ابدأ بورقة واحدة')}</h3><p>{tr('Upload a clear photo or use your camera.', 'ارفع صورة واضحة أو استخدم الكاميرا.')}</p></div>''', unsafe_allow_html=True)
        else:
            if st.button(tr('Analyze leaf →', 'تحليل الورقة ←'), type='primary', use_container_width=True):
                with st.spinner(tr('Examining your leaf…', 'جارٍ تحليل الورقة…')):
                    p, ms = predict(img)
                st.session_state.prediction = {'hash': hashlib.sha256(raw).hexdigest(), 'p': p, 'ms': ms}
            if st.button(tr('Start a new scan', 'بدء فحص جديد'), use_container_width=True):
                st.session_state.upload_id += 1
                st.session_state.prediction = None
                st.rerun()
    with right:
        st.markdown(f"### {tr('02 · Your plant insights', '02 · قراءة حالة النبات')}")
        result = st.session_state.prediction
        active = img is not None and result is not None and result['hash'] == hashlib.sha256(raw).hexdigest()
        if not active:
            st.markdown(f'''<div class="empty"><div class="empty-icon">◎</div><h3>{tr('A little insight goes a long way', 'خطوة صغيرة لفهم صحة نباتك')}</h3><p>{tr('Your result, confidence and closest matches will appear here after analysis.', 'بعد التحليل، ستظهر النتيجة ونسبة الثقة وأقرب التوقعات هنا.')}</p></div>''', unsafe_allow_html=True)
        else:
            p, ms = result['p'], result['ms']
            order = p.argsort()[::-1][:3]
            conf = float(p[order[0]])
            ent = float(-(p * np.log(p + 1e-9)).sum())
            crop, cond = pretty(labels[order[0]])
            state = 'uncertain' if conf < conf_t or ent > ent_t else ('healthy' if cond.lower() == 'healthy' else 'disease')
            color, tint, icon = {'healthy':('#368b59','#e3f3e6','✓'), 'disease':('#c45b42','#fae9e3','!'), 'uncertain':('#ab791b','#faf0d7','?')}[state]
            advice = t['adv_u'] if state == 'uncertain' else t['adv_d'] if state == 'disease' else tr('Keep monitoring your plant for visible changes.', 'استمر في مراقبة النبات لأي تغيّرات ظاهرة.')
            title = t['uncertain'] if state == 'uncertain' else crop
            subtitle = (' · '.join([t['why_c']] if conf < conf_t else []) + (' · ' if conf < conf_t and ent > ent_t else '') + (t['why_e'] if ent > ent_t else '')) if state == 'uncertain' else cond
            st.markdown(f'''<div class="card" style="--ink:{color};--tint:{tint};--angle:{conf*360:.2f}deg"><div class="kicker">{t['result']}</div><span class="badge">{icon} {t[state]}</span><div class="result-main"><div><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div><div class="ring"><div class="ring-center"><strong>{conf:.0%}</strong><small>{t['confl']}</small></div></div><div class="note">{advice}</div></div>''', unsafe_allow_html=True)
            rows = ''
            for k, i in enumerate(order):
                label = ' · '.join(pretty(labels[i]))
                rows += f'<div class="pred-row"><span>{k+1:02d} &nbsp; {esc(label)}</span><b>{float(p[i]):.1%}</b></div><div class="track"><div class="fill" style="width:{float(p[i])*100:.2f}%;background:{color if k==0 else "#7f9d8c"}"></div></div>'
            st.markdown(f'<div class="card"><div class="kicker">{t["top3"]}</div>{rows}</div>', unsafe_allow_html=True)
            with st.expander(t['details']):
                st.write(f"Entropy: {ent:.3f} / {ent_t:.2f}")
                st.write(f"{t['time']}: {ms:.0f} ms")
                st.caption(tr('Confidence is a model score, not a guarantee of correctness.', 'الثقة درجة يعطيها النموذج وليست ضمانًا لصحة النتيجة.'))
            report = {'crop':crop, 'condition':cond, 'confidence':round(conf,4), 'entropy':round(ent,4), 'status':state, 'confidence_threshold':conf_t, 'entropy_threshold':ent_t}
            st.download_button(tr('↓ Download result', '↓ تحميل النتيجة'), json.dumps(report, ensure_ascii=False, indent=2), 'crop_doctor_result.json', 'application/json', use_container_width=True)
            signature = (result['hash'], conf_t, ent_t)
            if signature not in st.session_state.seen:
                st.session_state.seen.add(signature)
                st.session_state.history.append({'time':datetime.now().isoformat(timespec='seconds'), **report})

with tab2:
    st.markdown(f"### {tr('Your growing journal', 'سجل العناية بنباتاتك')}")
    st.caption(tr('Session only. Each new image or threshold combination is recorded once.', 'سجل الجلسة الحالية فقط. تُحفظ كل صورة مع إعدادات حدودها مرة واحدة.'))
    if not st.session_state.history:
        st.info(t['hist_empty'])
    else:
        df = pd.DataFrame(st.session_state.history)
        cols = st.columns(3)
        for c, label, value in zip(cols, [tr('Recorded analyses','التحليلات المسجلة'),t['healthy'],t['uncertain']], [len(df),int((df.status=='healthy').sum()),int((df.status=='uncertain').sum())]):
            c.metric(label,value)
        search = st.text_input(tr('Search crop or condition','ابحث باسم المحصول أو الحالة'))
        status = st.selectbox(tr('Filter status','تصفية الحالة'), ['all','healthy','disease','uncertain'], format_func=lambda x: tr('All results','كل النتائج') if x=='all' else t[x])
        filtered = df.copy()
        if search:
            filtered = filtered[filtered.crop.str.contains(search,case=False,regex=False) | filtered.condition.str.contains(search,case=False,regex=False)]
        if status != 'all':
            filtered = filtered[filtered.status==status]
        st.dataframe(filtered.iloc[::-1], hide_index=True, use_container_width=True)
        st.download_button(t['dl'],filtered.to_csv(index=False).encode('utf-8-sig'),'crop_doctor_history.csv','text/csv')
        confirm = st.checkbox(tr('Confirm clearing session history','تأكيد مسح سجل الجلسة'))
        if st.button(t['clr'],disabled=not confirm):
            st.session_state.history = []
            # Keep signatures so the current visible result is not immediately re-added.
            st.rerun()
with tab3:
    st.markdown(f"### {t['how']}")
    st.write(t['how_t'])
    st.caption(tr('Metrics below are carried over from your original app; they have not been independently verified.', 'الأرقام التالية من التطبيق الأصلي، ولم تُتحقق بشكل مستقل.'))
    for c, key, value in zip(st.columns(4),['m1','m2','m3','m4'],['95.9%','2.58 MB','92.0%','98.7%']):
        c.metric(t[key],value)
    st.markdown(f"### {t['lim']}")
    for item in t['lim_t']:
        st.write(f'• {item}')
    st.info(tr('When hosted online, images are processed on the hosting server. Offline use applies to local installation with dependencies already installed.', 'عند استضافة التطبيق عبر الإنترنت، تُعالج الصور على خادم الاستضافة. العمل دون إنترنت يخص التشغيل المحلي بعد تثبيت المتطلبات.'))
st.markdown(f'<div class="footer">🌿 {t["by"]} · Crop Doctor</div>',unsafe_allow_html=True)
