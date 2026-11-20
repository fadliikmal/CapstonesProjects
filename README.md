# CapstonesProjects

This repository contains two applied machine learning capstone projects. Each
project is self-contained and includes its own source code, dependencies,
artifacts, and detailed documentation.

## Projects

### AIML-01: Real-Time Financial Fraud Detection

**Location:** [`AIML-01/fraud-detection/`](./AIML-01/fraud-detection/)

A production-oriented fraud detection platform for streaming financial
transactions. It uses an XGBoost model behind a FastAPI inference service and
provides an interactive Streamlit dashboard for operations and analyst review.

Key capabilities include:

- Low-latency transaction fraud scoring
- Three-tier risk decisions:
  - `LOW` → `APPROVE`
  - `MEDIUM` → `CHALLENGE_REVIEW`
  - `HIGH` → `AUTO_BLOCK`
- Real-time transaction replay through a stream producer simulator
- Analyst case review and transaction simulation dashboard
- Evidently AI data drift and prediction drift monitoring
- Local, Docker Compose, and Kubernetes/Minikube deployment options

Main components:

```text
AIML-01/fraud-detection/
├── app/          # FastAPI backend and risk engine
├── dashboard/    # Streamlit operations dashboard
├── streaming/    # Transaction replay producer
├── monitoring/   # Evidently AI drift monitoring
├── artifacts/    # Model and feature metadata
└── k8s/          # Kubernetes deployment manifests
```

See the [AIML-01 README](./AIML-01/fraud-detection/README.md) for installation,
API details, monitoring, and deployment instructions.

### AIML-04: Autonomous Perception Stack

**Location:** [`AIML-04/autonomous-perception-stack/`](./AIML-04/autonomous-perception-stack/)

An offline, CPU-oriented road-scene perception pipeline for video. It combines
object detection, multi-object tracking, lane perception, and monocular depth
estimation in an OpenCV application.

Key capabilities include:

- YOLO object detection with ONNX Runtime
- NumPy-based ByteTrack tracking with Kalman filter prediction
- TwinLiteNet lane segmentation with an OpenCV fallback
- MiDaS_small monocular depth estimation
- Estimated object distances and lead-object identification
- Lane offset and lane departure warning telemetry
- Annotated MP4 video output

Main components:

```text
AIML-04/autonomous-perception-stack/
├── main.py           # Video perception pipeline
├── src/              # Detection, depth, and lane modules
├── models/           # Local ONNX and other model artifacts
├── output/           # Generated videos
└── export_midas.py   # MiDaS_small ONNX export utility
```

The large `midas_small.onnx` model is intentionally excluded from GitHub.
After installing the project dependencies, generate it locally with:

```powershell
cd AIML-04/autonomous-perception-stack
python export_midas.py
```

See the [AIML-04 README](./AIML-04/autonomous-perception-stack/README.md) for
the complete setup, model, and execution instructions.

## Repository layout

```text
CapstonesProjects/
├── AIML-01/
│   └── fraud-detection/
├── AIML-04/
│   └── autonomous-perception-stack/
└── README.md
```

## Getting started

Choose a project above and follow its project-specific README. The projects
have separate Python environments and dependency manifests, so install
dependencies from the selected project directory rather than from the
repository root.

## Notes

- Large model files, generated videos, local virtual environments, and other
  generated artifacts may be excluded through project-level `.gitignore` rules.
- The AIML-01 platform is intended for fraud scoring and operational workflow
  demonstration.
- The AIML-04 pipeline is a perception prototype and is not a certified
  autonomous-driving control system.