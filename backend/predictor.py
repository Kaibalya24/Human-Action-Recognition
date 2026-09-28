import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model import CNNLSTM

from backend.config import (
    MODEL_PATH,
    SEQUENCE_LENGTH,
    CLASS_NAMES
)


class ActionPredictor:

    def __init__(self):

        # Use GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create the same CNN-LSTM architecture
        self.model = CNNLSTM(
            num_classes=len(CLASS_NAMES),
            hidden_size=128,
            num_layers=1
        )

        # Load trained model
        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        # Move model to selected device
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        # Same preprocessing used during training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []

        while len(frames) < SEQUENCE_LENGTH:

            ret, frame = cap.read()

            if not ret:
                break

            # OpenCV uses BGR
            # Convert to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Apply preprocessing
            frame = self.transform(frame)

            frames.append(frame)

        cap.release()

        # Make sure we have at least one frame
        if len(frames) == 0:
            raise ValueError(
                "Could not read any frames from the video."
            )

        # If video has fewer than 20 frames,
        # repeat the last frame
        while len(frames) < SEQUENCE_LENGTH:

            frames.append(frames[-1].clone())

        # Keep exactly 20 frames
        frames = frames[:SEQUENCE_LENGTH]

        # Shape:
        # [20, 3, 224, 224]
        frames = torch.stack(frames)

        # Add batch dimension:
        # [1, 20, 3, 224, 224]
        frames = frames.unsqueeze(0)

        return frames

    def predict(self, video_path):

        # Load video frames
        frames = self.load_video(video_path)

        # Move frames to CPU/GPU
        frames = frames.to(self.device)

        # Disable gradient calculation
        with torch.no_grad():

            # Model prediction
            outputs = self.model(frames)

            # Convert logits to probabilities
            probabilities = F.softmax(
                outputs,
                dim=1
            )

        # Get highest probability
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        predicted_index = predicted_index.item()
        confidence = confidence.item()

        # Convert probabilities to percentages
        probabilities = probabilities[0].cpu().numpy()

        probability_dict = {}

        for i, class_name in enumerate(CLASS_NAMES):

            probability_dict[class_name] = round(
                float(probabilities[i] * 100),
                2
            )

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(
                confidence * 100,
                2
            ),
            "probabilities": probability_dict
        }


# Create one predictor object
# so the model is loaded only once
predictor = ActionPredictor()