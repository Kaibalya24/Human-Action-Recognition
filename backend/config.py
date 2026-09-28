import os


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Trained model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_lstm_action_recognition.pth"
)


# Number of frames used by the CNN-LSTM model
SEQUENCE_LENGTH = 20


# Action classes
CLASS_NAMES = [
    "Walking with dog",
    "PullUps",
    "PushUps",
    "PizzaTossing"
]