import os
import random
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import VideoDataset
from model import CNNLSTM


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


CLASS_NAMES = [
    "Walking with dog",
    "PullUps",
    "PushUps",
    "PizzaTossing"
]
NUM_CLASSES = len(CLASS_NAMES)

SEQUENCE_LENGTH = 20

BATCH_SIZE = 2

EPOCHS = 25

LEARNING_RATE = 0.0001


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 3. FIND ALL VIDEOS
# ============================================================

video_paths = []
labels = []


print("\nSearching for videos...\n")


for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(
        DATASET_DIR,
        class_name
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


# ============================================================
# 4. DISPLAY DATASET INFORMATION
# ============================================================

print("Total videos found:", len(video_paths))

for index, class_name in enumerate(CLASS_NAMES):

    count = labels.count(index)

    print(
        f"{class_name}: {count} videos"
    )


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


print("\nTraining videos:", len(train_paths))
print("Validation videos:", len(val_paths))


# ============================================================
# 6. CREATE DATASETS
# ============================================================

train_dataset = VideoDataset(
    train_paths,
    train_labels,
    sequence_length=SEQUENCE_LENGTH
)

val_dataset = VideoDataset(
    val_paths,
    val_labels,
    sequence_length=SEQUENCE_LENGTH
)


# ============================================================
# 7. CREATE DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)


# ============================================================
# 9. CREATE MODEL
# ============================================================

model = CNNLSTM(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# 10. LOSS FUNCTION
# ============================================================

criterion = torch.nn.CrossEntropyLoss()


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# ============================================================
# 12. STORE RESULTS
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# 13. TRAINING LOOP
# ============================================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for videos, labels_batch in train_loader:

        videos = videos.to(device)

        labels_batch = labels_batch.to(device)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(videos)


        # Calculate loss
        loss = criterion(
            outputs,
            labels_batch
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()


    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for videos, labels_batch in val_loader:

            videos = videos.to(device)

            labels_batch = labels_batch.to(device)


            outputs = model(videos)


            loss = criterion(
                outputs,
                labels_batch
            )


            val_running_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels_batch.size(0)

            val_correct += (
                predicted == labels_batch
            ).sum().item()


    val_loss = (
        val_running_loss / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)

    val_accuracies.append(val_accuracy)


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "cnn_lstm_action_recognition.pth"
)


torch.save(
    model.state_dict(),
    model_path
)


print("\nModel saved to:")

print(model_path)


# ============================================================
# 15. PLOT LOSS
# ============================================================

plt.figure()

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "CNN-LSTM Training and Validation Loss"
)

plt.legend()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )
)

plt.close()


# ============================================================
# 16. PLOT ACCURACY
# ============================================================

plt.figure()

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "CNN-LSTM Training and Validation Accuracy"
)

plt.legend()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    )
)

plt.close()


print("\nTraining graphs saved!")

print(
    "\nTraining completed successfully!"
)