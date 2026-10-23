import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

IMG = 224
CONF_THRESHOLD = 0.80
ENTROPY_MAX = 1.0

st.set_page_config(page_title="Crop Doctor", page_icon="🌿")
st.title("🌿 Crop Disease Detector")
st.caption("Runs a compact model with no internet needed for inference. "
           "It will say 'uncertain' instead of guessing.")

@st.cache_resource
def load():
    it = tf.lite.Interpreter(model_path="model.tflite")
    it.allocate_tensors()
    return it, json.load(open("labels.json"))

interp, labels = load()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

def predict(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG, IMG))
    x = np.asarray(img, dtype="float32")[None]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    return interp.get_tensor(out["index"])[0]

def pretty(name):
    crop, cond = name.split("___")
    return crop.replace("_", " "), cond.replace("_", " ")

src = st.radio("Image source", ["Camera", "Upload"], horizontal=True)
file = st.camera_input("Take a photo of one leaf") if src == "Camera" \
    else st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    p = predict(img)
    order = p.argsort()[::-1][:3]
    conf = float(p[order[0]])
    ent = float(-(p * np.log(p + 1e-9)).sum())

    if conf < CONF_THRESHOLD or ent > ENTROPY_MAX:
        st.warning(f"⚠️ Uncertain (confidence {conf:.0%}). "
                   "Retake the photo (single leaf, good light, fill the frame) "
                   "or ask a local agricultural expert.")
    else:
        crop, cond = pretty(labels[order[0]])
        if cond.lower() == "healthy":
            st.success(f"✅ {crop}: looks healthy ({conf:.0%})")
        else:
            st.error(f"🦠 {crop}: {cond} ({conf:.0%})")
            st.info("Confirm with an extension officer before applying any treatment.")

    with st.expander("Top-3 predictions"):
        for i in order:
            c, d = pretty(labels[i])
            st.write(f"{c} - {d}: {p[i]:.1%}")
st.divider()
st.caption("Trained on PlantVillage (lab-style images). Accuracy on field photos may be lower.")
