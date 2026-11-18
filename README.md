# SoundSense 🎧

**ESC-50 Environmental Sound Classification Platform & Scientific Studio**

SoundSense is an end-to-end environmental sound classification system featuring deep learning model inference (ResNet-18, VGG-16, SResNet, CNN-GRU), acoustic signal processing (Log-Mel Spectrograms, MFCCs, and Threshold-Free Recurrence Plots), a responsive dark-mode web application, and developer training workflows.

---

## 🌟 Features

- **Classifier Studio**:
  - Drag-and-drop audio file upload (WAV, MP3, OGG, FLAC, M4A, AAC)
  - Live microphone recording with real-time audio visualization
  - 1-click test presets with ESC-50 sample audio
  - Real-time confidence gauge, Top-5 softmax probability ranking, and acoustic telemetry (RMS, ZCR, dBFS, latency)
- **Scientific Acoustic Visualizer**:
  - 128-band Log-Mel Spectrogram
  - 128×128 Threshold-Free Recurrence Plot (RP) for non-linear phase space dynamics
  - Energy-based active segment (VAD) detection
- **Model Architectures**:
  - ResNet-18 (Deep Residual Network)
  - VGG-16 (16-layer Deep Convolutional Network)
  - SResNet (Squeezed Residual Network)
  - CNN-GRU (Convolutional Recurrent Neural Network)
- **Developer Management**:
  - Hyperparameter tuning and model training runs
  - Checkpoint registry and model hot-swapping
  - System telemetry and dataset analytics
  - JWT-based authentication with role-based access control (User / Developer)

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js (optional, Express fallback)

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/rohanputta77/SoundSense.git
cd SoundSense

# Install required Python dependencies
pip install -r requirements.txt
```

### 3. Run the Server

```bash
# Navigate to the application directory
cd Sound_Recognition_1

# Start the FastAPI server
python main.py
```

The application will be live at:
- **Web UI**: [http://localhost:3001](http://localhost:3001)
- **API Documentation (Swagger)**: [http://localhost:3001/docs](http://localhost:3001/docs)
- **Health Check**: [http://localhost:3001/api/health](http://localhost:3001/api/health)

---

## 📁 Repository Structure

```text
SoundSense/
├── requirements.txt               # Project Python dependencies
├── README.md                      # Documentation
├── Sound_Recognition_1/
│   ├── main.py                    # FastAPI server & route orchestration
│   ├── auth.py                    # JWT authentication & user roles
│   ├── server.js                  # Express.js server alternative
│   ├── package.json               # Node.js manifest
│   ├── Front-End/                 # Interactive web interface
│   │   ├── index.html             # Single-page UI
│   │   ├── app.js                 # Frontend application logic & audio visualization
│   │   ├── styles.css             # Glassmorphism styling & design system
│   │   └── samples/               # Audio sample clips for instant testing
│   ├── Preproccesing/             # Acoustic signal processing & deep learning models
│   │   ├── acoustic_inference.py  # Model inference pipeline
│   │   ├── audio_processor.py     # Resampling, normalization, VAD, PAA
│   │   ├── feature_extractor.py   # Log-Mel, MFCC, and Recurrence Plot extractors
│   │   ├── models.py              # PyTorch model definitions
│   │   └── download_checkpoints.py# Hugging Face checkpoint retriever
│   ├── checkpoints/               # Model checkpoints & manifest
│   ├── weights/                   # Trained model weights (.pth)
│   └── uploads/                   # Temporary audio upload buffer
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `POST` | `/api/predict` | Classify uploaded audio buffer | Public / Authenticated |
| `POST` | `/api/auth/signup` | Register a new user | Public |
| `POST` | `/api/auth/login` | Authenticate & retrieve JWT | Public |
| `GET`  | `/api/me` | Fetch active user profile & role | Authenticated |
| `GET`  | `/api/history` | Retrieve personal prediction logs | Authenticated |
| `POST` | `/api/train` | Trigger model training run | Developer |
| `POST` | `/api/save-model` | Save model checkpoint | Developer |
| `GET`  | `/api/stats` | System telemetry & dataset metrics | Developer |
| `GET`  | `/api/health` | Service health status | Public |

---

## 📜 License
Apache-2.0
