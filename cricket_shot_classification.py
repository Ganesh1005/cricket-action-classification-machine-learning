import os
import cv2
import json
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, BatchNormalization,
                                     GlobalAveragePooling3D, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns

# ---- config/dataclass ----
@dataclass
class Config:
    frame_height: int = 100
    frame_width: int = 100
    num_frames: int = 14
    num_classes: int = 5
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-4
    seed: int = 42

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ==================== VIDEO PROCESSING ====================
class VideoProcessor:
    def __init__(self, frame_height: int, frame_width: int, num_frames: int):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_frame_indices(total_frames, self.num_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32))
        cap.release()

        frames = np.array(frames)
        if len(frames) < self.num_frames:
            padding = np.zeros((self.num_frames - len(frames), self.frame_height, self.frame_width, 3), dtype=np.float32)
            frames = np.concatenate([frames, padding])
        return frames[:self.num_frames]

    def _get_frame_indices(self, total_frames, num_frames):
        if total_frames <= num_frames:
            return list(range(total_frames))
        step = max(1, total_frames // num_frames)
        return [min(i * step, total_frames - 1) for i in range(num_frames)]

# ==================== DATA LOADING ====================
class CricketDataset:
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.class_names = []

    def load_dataset(self, data_path: Path, expected_shape):
        video_paths, labels = [], []
        self.class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

        for class_idx, class_name in enumerate(self.class_names):
            class_path = data_path / class_name
            for video_file in class_path.glob("*.mp4"):
                video_paths.append(video_file)
                labels.append(class_idx)

        X, y = [], []
        for vp, label in zip(video_paths, labels):
            try:
                frames = self.processor.extract_frames(vp)
                if frames.shape == expected_shape:
                    X.append(frames)
                    y.append(label)
                else:
                    print(f"Skipping {vp} - invalid shape: {frames.shape}")
            except Exception as e:
                print(f"Error processing {vp}: {e}")

        if not X:
            raise ValueError(f"No valid videos found in {data_path}")
        return np.array(X, dtype=np.float32), np.array(y)

# ==================== MODEL ARCHITECTURE ====================
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((1, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# ==================== UTILS ====================
def save_confusion_matrix(cm, class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ==================== MAIN ====================
def parse_args():
    p = argparse.ArgumentParser(description="Train 3D CNN for cricket shot classification")
    p.add_argument("--train_dir", type=str, required=True, help="Path to training data (class subfolders with .mp4)")
    p.add_argument("--test_dir", type=str, default="", help="Path to test data (optional)")
    p.add_argument("--outputs_dir", type=str, default="outputs", help="Where to save models/plots/logs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--frame_h", type=int, default=100)
    p.add_argument("--frame_w", type=int, default=100)
    p.add_argument("--num_frames", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(frame_height=args.frame_h, frame_width=args.frame_w, num_frames=args.num_frames,
                 num_classes=args.num_classes, batch_size=args.batch_size, epochs=args.epochs,
                 learning_rate=args.lr, seed=args.seed)

    set_seeds(cfg.seed)

    outputs_dir = Path(args.outputs_dir)
    models_dir = outputs_dir / "models"
    plots_dir = outputs_dir / "plots"
    logs_dir = outputs_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    with open(outputs_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg) | {"train_dir": args.train_dir, "test_dir": args.test_dir}, f, indent=2)

    # Data
    processor = VideoProcessor(cfg.frame_height, cfg.frame_width, cfg.num_frames)
    dataset = CricketDataset(processor)

    expected_shape = (cfg.num_frames, cfg.frame_height, cfg.frame_width, 3)

    X, y = dataset.load_dataset(Path(args.train_dir), expected_shape)
    y = to_categorical(y, num_classes=cfg.num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=np.argmax(y, axis=1)
    )

    # Model
    model = build_model(
        input_shape=(cfg.num_frames, cfg.frame_height, cfg.frame_width, 3),
        num_classes=cfg.num_classes
    )
    model.compile(optimizer=Adam(learning_rate=cfg.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Training
    best_model_path = models_dir / "cricket_shot_3d_attention_model4.h5"
    callbacks = [
        ModelCheckpoint(best_model_path.as_posix(), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Optional test
    if args.test_dir and Path(args.test_dir).exists():
        X_test, y_test_int = dataset.load_dataset(Path(args.test_dir), expected_shape)
        y_test = to_categorical(y_test_int, num_classes=cfg.num_classes)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc*100:.2f}%")

        y_pred = model.predict(X_test)
        y_true = y_test_int
        y_pred_classes = np.argmax(y_pred, axis=1)

        # report
        report = classification_report(y_true, y_pred_classes, target_names=dataset.class_names, output_dict=True)
        with open(outputs_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        save_confusion_matrix(cm, dataset.class_names, plots_dir / "confusion_matrix.png")
    else:
        print("Test dir not provided or not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
