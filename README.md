# Crop Disease Detector
Offline, uncertainty-aware leaf disease classifier (MobileNetV2 -> TFLite, PlantVillage, 38 classes).

## Run
1. `pip install -r requirements-train.txt && python train.py` (GPU recommended, or use `crop_disease_detector.ipynb` on Colab). Produces `deploy/model.tflite`, `deploy/labels.json`, `metrics.json`.
2. Local app: `cd deploy && pip install -r requirements.txt && streamlit run app.py`
3. Docker: `cd deploy && docker build -t crop-doctor . && docker run -p 7860:7860 crop-doctor`
4. Hugging Face Spaces: new Docker Space, upload `deploy/` contents.
5. Android: use `model.tflite` + `labels.json` (input 224x224 float32, 0-255).

## Structure
`train.py` training/eval/export | `deploy/` app, Dockerfile, model | `REPORT.md` project report
