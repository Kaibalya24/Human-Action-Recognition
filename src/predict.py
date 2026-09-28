import os
import sys

import cv2
import torch
from torchvision import transforms

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

SEQUENCE_LENGTH = 20

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_lstm_action_recognition.pth"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD VIDEO FRAMES
# ============================================================

def load_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while len(frames) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV: BGR
        # Convert to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = transform(frame)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            f"Could not read video: {video_path}"
        )

    # If fewer than 20 frames,
    # repeat the last frame
    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            frames[-1].clone()
        )

    # Keep exactly 20 frames
    frames = frames[:SEQUENCE_LENGTH]

    # Convert list to tensor
    frames = torch.stack(frames)

    # Add batch dimension
    frames = frames.unsqueeze(0)

    return frames


# ============================================================
# PREDICTION
# ============================================================

def predict(video_path):

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing device:", device)

    # Check model exists
    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Create model
    model = CNNLSTM(
        num_classes=len(CLASS_NAMES)
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    # Evaluation mode
    model.eval()

    print("Model loaded successfully!")

    # Load video
    print("Reading video...")

    frames = load_video(video_path)

    frames = frames.to(device)

    print(
        "Video tensor shape:",
        frames.shape
    )

    # Prediction
    with torch.no_grad():

        outputs = model(frames)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    predicted_action = CLASS_NAMES[
        predicted_index
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # Display result
    print("\n" + "=" * 50)

    print(
        "Predicted Action:",
        predicted_action
    )

    print(
        f"Confidence: {confidence_value:.2f}%"
    )

    print("=" * 50)

    print("\nClass probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):

        probability = (
            probabilities[0][index].item()
            * 100
        )

        print(
            f"{class_name}: {probability:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src\\predict.py <video_path>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src\\predict.py test_videos\\test.mp4"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print(
            f"\nVideo not found: {video_path}"
        )

        sys.exit(1)

    predict(video_path)