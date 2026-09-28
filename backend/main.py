import os
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import predictor


# Create FastAPI application
app = FastAPI(
    title="Human Action Recognition API",
    description="CNN-LSTM based Human Action Recognition",
    version="1.0.0"
)


# Allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Temporary folder for uploaded videos
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEMP_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "temp"
)

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)


# Home / health-check endpoint
@app.get("/")
def root():

    return {
        "message": "Human Action Recognition API is running"
    }


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check that a file was actually uploaded
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file was uploaded."
        )

    # Get file extension
    extension = os.path.splitext(
        file.filename
    )[1].lower()

    # Allowed video formats
    allowed_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    ]

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Please upload MP4, AVI, MOV, or MKV."
            )
        )

    # Generate a unique temporary filename
    filename = (
        uuid.uuid4().hex
        + extension
    )

    video_path = os.path.join(
        TEMP_DIR,
        filename
    )

    try:

        # Read uploaded video
        video_data = await file.read()

        # Save temporarily
        with open(
            video_path,
            "wb"
        ) as video_file:

            video_file.write(video_data)

        # Run model prediction
        result = predictor.predict(
            video_path
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        # Delete temporary video
        if os.path.exists(video_path):

            os.remove(video_path)