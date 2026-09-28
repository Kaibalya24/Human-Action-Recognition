import torch
from model import CNNLSTM


def main():

    print("Creating CNN-LSTM model...")

    model = CNNLSTM(
        num_classes=4
    )

    print("Model created successfully!")

    # Fake input:
    #
    # 2 videos
    # 20 frames each
    # 3 RGB channels
    # 224 x 224 pixels

    dummy_input = torch.randn(
        2,
        20,
        3,
        224,
        224
    )

    print(
        "Input shape:",
        dummy_input.shape
    )

    # Forward pass

    output = model(dummy_input)

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Model test successful!"
    )


if __name__ == "__main__":

    main()