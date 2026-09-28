import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):

    def __init__(self, video_paths, labels, sequence_length=20):

        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length

        # Preprocessing for ResNet18
        self.transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):

        return len(self.video_paths)

    def __getitem__(self, index):

        video_path = self.video_paths[index]
        label = self.labels[index]

        # Open video
        cap = cv2.VideoCapture(video_path)

        frames = []

        # Read frames
        while len(frames) < self.sequence_length:

            ret, frame = cap.read()

            if not ret:
                break

            # OpenCV uses BGR
            # Convert BGR → RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize + normalize
            frame = self.transform(frame)

            frames.append(frame)

        cap.release()

        # Make sure at least one frame was read
        if len(frames) == 0:

            raise RuntimeError(
                f"Could not read video: {video_path}"
            )

        # If video has fewer than 20 frames,
        # repeat the last frame
        while len(frames) < self.sequence_length:

            frames.append(
                frames[-1].clone()
            )

        # Keep exactly 20 frames
        frames = frames[:self.sequence_length]

        # Convert list → tensor
        frames = torch.stack(frames)

        return (
            frames,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )