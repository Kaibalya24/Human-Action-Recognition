import os

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from dataset import VideoDataset
from model import CNNLSTM


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "Walking with dog",
    "PullUps",
    "PushUps",
    "PizzaTossing"
]

NUM_CLASSES = len(CLASS_NAMES)

SEQUENCE_LENGTH = 20
BATCH_SIZE = 2

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_lstm_action_recognition.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# FIND VIDEOS
# ============================================================

video_paths = []
labels = []

print("\nSearching for videos...\n")

for class_index, class_name in enumerate(CLASS_NAMES):

    # Handle the actual folder names
    # used in your dataset
    folder_names = {
        "Walking with dog": "Walking with dog",
        "PullUps": "PullUps",
        "PushUps": "PushUps",
        "PizzaTossing": "PizzaTossing"
    }

    class_folder = os.path.join(
        DATASET_DIR,
        folder_names[class_name]
    )

    if not os.path.exists(class_folder):

        raise FileNotFoundError(
            f"Dataset folder not found: {class_folder}"
        )

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):

            video_path = os.path.join(
                class_folder,
                filename
            )

            video_paths.append(video_path)
            labels.append(class_index)


print(
    "Total videos found:",
    len(video_paths)
)


for index, class_name in enumerate(CLASS_NAMES):

    count = labels.count(index)

    print(
        f"{class_name}: {count} videos"
    )


# ============================================================
# CREATE SAME VALIDATION SPLIT
# ============================================================

_, val_paths, _, val_labels = train_test_split(
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


print(
    "\nValidation videos:",
    len(val_paths)
)


# ============================================================
# DATASET
# ============================================================

val_dataset = VideoDataset(
    val_paths,
    val_labels,
    sequence_length=SEQUENCE_LENGTH
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nUsing device:",
    device
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = CNNLSTM(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

print("\nEvaluating validation videos...\n")

with torch.no_grad():

    for videos, labels_batch in val_loader:

        videos = videos.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(videos)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)

print("MODEL EVALUATION RESULTS")

print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1-Score : {f1 * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("Confusion Matrix:\n")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

display_labels = [
    "Walking",
    "PullUps",
    "PushUps",
    "Pizza"
]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "CNN-LSTM Confusion Matrix"
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved to:"
)

print(
    confusion_matrix_path
)

print(
    "\nEvaluation completed successfully!"
)