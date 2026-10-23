"""Train, evaluate and export the crop disease classifier. Usage: python train.py"""
import json, os, numpy as np, tensorflow as tf, tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, f1_score

IMG, BATCH, SEED, THRESH = 224, 32, 42, 0.80
tf.keras.utils.set_random_seed(SEED)
(tr, va, te), info = tfds.load("plant_village", split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                               as_supervised=True, with_info=True)
names = info.features["label"].names
prep = lambda x, y: (tf.cast(tf.image.resize(x, (IMG, IMG)), tf.float32), y)
A = tf.data.AUTOTUNE
tr = tr.map(prep, A).shuffle(2000, seed=SEED).batch(BATCH).prefetch(A)
va = va.map(prep, A).batch(BATCH).prefetch(A)
te = te.map(prep, A).batch(BATCH).prefetch(A)

base = tf.keras.applications.MobileNetV2((IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False
i = layers.Input((IMG, IMG, 3))
x = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.15),
                         layers.RandomZoom(0.2), layers.RandomContrast(0.2)])(i)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)
x = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(base(x, training=False)))
model = models.Model(i, layers.Dense(len(names), activation="softmax")(x))
cb = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy")]
comp = lambda lr: model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
comp(1e-3); model.fit(tr, validation_data=va, epochs=5, callbacks=cb)
base.trainable = True
for l in base.layers[:-30]: l.trainable = False
comp(1e-5); model.fit(tr, validation_data=va, epochs=5, callbacks=cb)

y = np.concatenate([b.numpy() for _, b in te]); p = model.predict(te, verbose=0)
yp, conf = p.argmax(1), p.max(1); m = conf >= THRESH
metrics = {"test_acc": float((y == yp).mean()), "macro_f1": float(f1_score(y, yp, average="macro")),
           "threshold": THRESH, "coverage": float(m.mean()), "acc_on_answered": float((y[m] == yp[m]).mean())}
print(classification_report(y, yp, target_names=names, digits=3)); print(metrics)

os.makedirs("deploy", exist_ok=True)
c = tf.lite.TFLiteConverter.from_keras_model(model); c.optimizations = [tf.lite.Optimize.DEFAULT]
open("deploy/model.tflite", "wb").write(c.convert())
metrics["tflite_mb"] = os.path.getsize("deploy/model.tflite") / 1e6
json.dump(names, open("deploy/labels.json", "w")); json.dump(metrics, open("metrics.json", "w"), indent=2)
