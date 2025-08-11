!pip install opencv-python tensorflow keras-tuner

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D,
                                    BatchNormalization, GlobalAveragePooling3D,
                                    Dense, Dropout, Multiply, Reshape,
                                    Concatenate, Add, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                       ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kerastuner import HyperModel, RandomSearch
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

FRAME_HEIGHT = 100
FRAME_WIDTH = 100
NUM_FRAMES = 14
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4

# Paths
TRAIN_DATA_PATH = "/content/drive/MyDrive/CKT Dataset"
TEST_DATA_PATH = "/content/drive/MyDrive/test videos3"
MODEL_PATH = "/content/drive/MyDrive/cricket_shot_3d_attention_model4.h5"


class VideoProcessor:
    def __init__(self, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def extract_frames(self, video_path, num_frames=NUM_FRAMES):
        """Extracts and processes frames with guaranteed consistent shape"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # To Get frame indices with fallback for short videos
        frame_indices = self._get_frame_indices(total_frames, num_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = self._process_frame(frame)
                frames.append(frame)
            else:
                # Use black frame if reading fails
                frames.append(np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32))

        cap.release()

        # Ensure exactly NUM_FRAMES frames
        frames = np.array(frames)
        if len(frames) < num_frames:
            # Pad with black frames for missing frames
            padding = np.zeros((num_frames - len(frames), self.frame_height, self.frame_width, 3), dtype=np.float32)
            frames = np.concatenate([frames, padding])

        return frames[:num_frames]

    def _process_frame(self, frame):
        """Process individual frame with consistent output shape"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        frame = frame / 255.0  # Normalize
        return frame

    def _get_frame_indices(self, total_frames, num_frames):
        """Generate valid frame indices with fallback for short videos"""
        if total_frames <= num_frames:
            return list(range(total_frames))

        step = max(1, total_frames // num_frames)
        return [min(i * step, total_frames - 1) for i in range(num_frames)]

# DATA LOADING
class CricketDataset:
    def __init__(self, processor):
        self.processor = processor
        self.class_names = []

    def load_dataset(self, data_path):
        """Load and process dataset with strict shape validation"""
        video_paths = []
        labels = []
        self.class_names = sorted(os.listdir(data_path))

        # First pass: collect all video paths
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                continue

            print(f"Processing class: {class_name}")
            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    video_paths.append(os.path.join(class_path, video_file))
                    labels.append(class_idx)

        # Second pass: process videos with strict shape validation
        X = []
        y = []
        expected_shape = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)

        for video_path, label in zip(video_paths, labels):
            try:
                frames = self.processor.extract_frames(video_path)

                # Strict shape validation
                if frames.shape == expected_shape:
                    X.append(frames)
                    y.append(label)
                else:
                    print(f"Skipping {video_path} - invalid shape: {frames.shape}")

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")

        if not X:
            raise ValueError("No valid videos found after processing")

        return np.array(X, dtype=np.float32), np.array(y)

# MODEL ARCHITECTURE
class CricketShotModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """Build 3D CNN model with simplified architecture"""
        inputs = Input(shape=self.input_shape)

        # Conv3D layers
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling3D((1, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = BatchNormalization()(x)

        # Global pooling and dense layers
        x = GlobalAveragePooling3D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model

# MAIN EXECUTION
def main():
    # Initialize components
    processor = VideoProcessor()
    dataset = CricketDataset(processor)

    # Load and prepare data
    print("\nLoading training data...")
    try:
        X, y = dataset.load_dataset(TRAIN_DATA_PATH)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    # Encode labels
    y = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build model
    print("\nBuilding model...")
    model_builder = CricketShotModel(
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3),
        num_classes=NUM_CLASSES
    )
    model = model_builder.build_model()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')
        ],
        verbose=1
    )

    # Evaluate on test data if available
    if os.path.exists(TEST_DATA_PATH):
        print("\nLoading test data...")
        try:
            X_test, y_test = dataset.load_dataset(TEST_DATA_PATH)
            y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

            # Evaluation
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"\nTest Accuracy: {accuracy*100:.2f}%")

            # Detailed metrics
            y_pred = model.predict(X_test)
            y_true = np.argmax(y_test, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred_classes, target_names=dataset.class_names))

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=dataset.class_names,
                        yticklabels=dataset.class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        except Exception as e:
            print(f"Error evaluating test xdata: {e}")
    else:
        print("\nTest path not found. Skipping evaluation.")

if __name__ == "__main__":
    main()
