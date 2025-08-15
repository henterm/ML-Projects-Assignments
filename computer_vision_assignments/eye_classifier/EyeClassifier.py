import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class OpenEyesTrainer:
    def __init__(self, use_pseudo_labels=True, image_size=(224, 224)):
        self.use_pseudo_labels = use_pseudo_labels
        self.image_size = image_size
        self.model = None
        self.val_paths = []
        self.y_val = []

        self.labeled_paths = glob("labeled_self_train/labeled_self_train/**/*.jpg", recursive=True)
        self.labeled_labels = [self.get_label(p) for p in self.labeled_paths]

        self.pseudo_paths = []  # Добавляются через set_pseudo_labels()
        self.pseudo_labels = []

    def set_pseudo_labels(self, paths, labels):
        self.pseudo_paths = list(paths)
        self.pseudo_labels = list(labels)

    def get_label(self, path):
        return 1 if os.path.basename(os.path.dirname(path)) == "open" else 0

    def load_images(self, paths):
        X = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[WARNING] Can't read: {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size).astype(np.float32) / 255.0
            X.append(img)
        return np.stack(X)


    def image_generator(paths, labels, image_size, batch_size):
        def load_img(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def build_model(self):
        base = MobileNetV2(input_shape=(*self.image_size, 3), include_top=False, weights="imagenet")
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base.input, outputs=out)
        for layer in base.layers:
            layer.trainable = False
        model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, epochs=30, batch_size=32):
        # Объединение размеченных и псевдоразмеченных
        all_paths = self.labeled_paths.copy()
        all_labels = self.labeled_labels.copy()
        if self.use_pseudo_labels and self.pseudo_paths:
            all_paths += self.pseudo_paths
            all_labels += self.pseudo_labels

        y_all = np.array(all_labels, dtype=np.int32)

        paths_train, paths_val, y_train, y_val = train_test_split(
            all_paths, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        self.val_paths = paths_val  # сохраняем для оценки
        self.y_val = y_val

        X_train = self.load_images(paths_train)
        X_val = self.load_images(paths_val)

        self.model = self.build_model()
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )

        print("[INFO] Обучение завершено.")

    def predict_proba(self, path):
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Can't read image: {path}")
            return 0.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return float(self.model.predict(img, verbose=0)[0, 0])

    def compute_eer(self, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        idx = np.nanargmin(abs_diffs)
        eer = fpr[idx]
        threshold = thresholds[idx]
        return eer, threshold

    def evaluate(self):
        if self.model is None or not self.val_paths:
            print("[ERROR] Модель не обучена или нет валидационного сета")
            return

        y_scores = [self.predict_proba(p) for p in self.val_paths]
        eer, thr = self.compute_eer(self.y_val, y_scores)

        y_pred = [1 if s >= thr else 0 for s in y_scores]
        acc = accuracy_score(self.y_val, y_pred)
        print(f"\n[RESULTS] EER: {eer:.4f}, Threshold: {thr:.4f}, Accuracy: {acc:.4f}")
        print(classification_report(self.y_val, y_pred))

        # ROC-кривая
        fpr, tpr, _ = roc_curve(self.y_val, y_scores)
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, path="open_eyes_model.h5"):
        if self.model:
            self.model.save_weights(path)
            print(f"[INFO] Сохранено: {path}")