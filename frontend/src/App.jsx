import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [video, setVideo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleVideoChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setVideo(file);
    setResult(null);
    setError("");

    const videoURL = URL.createObjectURL(file);
    setPreview(videoURL);
  };

  const handlePredict = async () => {
    if (!video) {
      setError("Please select a video first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();

    formData.append("file", video);

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      setResult(data);

    } catch (err) {
      setError(err.message);

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      <header className="header">
        <h1>Human Action Recognition</h1>

        <p>
          CNN + LSTM Video Action Classifier
        </p>
      </header>

      <main className="container">

        <section className="upload-card">

          <h2>Upload Your Video</h2>

          <p className="description">
            Upload a short video and let the CNN-LSTM
            model identify the human action.
          </p>

          <label className="upload-box">

            <input
              type="file"
              accept="video/*"
              onChange={handleVideoChange}
            />

            <span className="upload-icon">
              📹
            </span>

            <span className="upload-text">
              {video
                ? video.name
                : "Choose a video file"}
            </span>

            <span className="upload-hint">
              MP4, AVI, MOV or MKV
            </span>

          </label>


          {preview && (
            <div className="video-preview">

              <h3>Video Preview</h3>

              <video
                src={preview}
                controls
                playsInline
              />

            </div>
          )}


          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={!video || loading}
          >

            {loading
              ? "Predicting..."
              : "Predict Action"}

          </button>


          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          {result && (
            <section className="result-card">

              <h2>Prediction Result</h2>

              <div className="prediction">

                <span className="prediction-label">
                  Predicted Action
                </span>

                <span className="prediction-value">
                  {result.prediction}
                </span>

              </div>


              <div className="confidence">

                <span>
                  Confidence
                </span>

                <strong>
                  {result.confidence}%
                </strong>

              </div>


              <div className="probabilities">

                <h3>
                  Class Probabilities
                </h3>

                {Object.entries(
                  result.probabilities
                ).map(
                  ([className, probability]) => (

                    <div
                      className="probability-row"
                      key={className}
                    >

                      <div className="probability-info">

                        <span>
                          {className}
                        </span>

                        <span>
                          {probability}%
                        </span>

                      </div>

                      <div className="progress-bar">

                        <div
                          className="progress"
                          style={{
                            width: `${probability}%`,
                          }}
                        />

                      </div>

                    </div>

                  )
                )}

              </div>

            </section>
          )}

        </section>

      </main>

    </div>
  );
}

export default App;