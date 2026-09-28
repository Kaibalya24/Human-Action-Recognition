import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes=4,
        hidden_size=128,
        num_layers=1
    ):

        super(CNNLSTM, self).__init__()

        # ------------------------------------------------
        # CNN PART
        # ------------------------------------------------

        # Load pretrained ResNet18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove ResNet's final classification layer
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Freeze CNN parameters
        # This makes training faster
        for parameter in self.cnn.parameters():

            parameter.requires_grad = False

        # ResNet18 produces 512 features
        self.feature_size = 512

        # ------------------------------------------------
        # LSTM PART
        # ------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # ------------------------------------------------
        # CLASSIFICATION PART
        # ------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # Input:
        #
        # batch_size
        # sequence_length
        # channels
        # height
        # width

        batch_size, sequence_length, channels, height, width = x.shape

        # ------------------------------------------------
        # Prepare input for CNN
        # ------------------------------------------------

        x = x.view(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        # ------------------------------------------------
        # CNN feature extraction
        # ------------------------------------------------

        features = self.cnn(x)

        # ResNet output:
        #
        # [batch × sequence, 512, 1, 1]

        features = features.view(
            batch_size,
            sequence_length,
            self.feature_size
        )

        # Now:
        #
        # [batch, sequence, 512]

        # ------------------------------------------------
        # LSTM temporal processing
        # ------------------------------------------------

        lstm_output, _ = self.lstm(features)

        # Take output from final frame
        final_output = lstm_output[:, -1, :]

        # ------------------------------------------------
        # Final classification
        # ------------------------------------------------

        output = self.fc(final_output)

        return output