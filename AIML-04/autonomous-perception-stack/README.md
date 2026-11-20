# Autonomous Perception Stack

An offline, CPU-oriented perception pipeline for road-scene video. The application
combines object detection, multi-object tracking, lane perception, and monocular
depth estimation in a single OpenCV display and can save the annotated result as
an MP4 video.

## Features

- **Object detection:** YOLO exported to ONNX Runtime.
- **Object tracking:** a lightweight NumPy ByteTrack implementation with Kalman
  filter prediction on skipped frames.
- **Lane perception:** TwinLiteNet ONNX segmentation with polynomial lane fitting.
  If the TwinLiteNet model is unavailable, the application falls back to the
  OpenCV lane detector.
- **Depth estimation:** Intel MiDaS_small exported to ONNX and executed in a
  separate process so depth inference does not block the main video loop.
- **Distance and lead-object estimation:** combines MiDaS relative depth with
  camera geometry and known object heights.
- **Runtime overlay:** object IDs, classes, estimated distances, lane status,
  lateral offset, departure warnings, FPS, and an optional depth preview.

## Project structure

```text
autonomous-perception-stack/
├── main.py                 # Video pipeline entry point
├── export_midas.py         # Downloads and exports MiDaS to ONNX
├── requirements.txt        # Python dependencies
├── models/
│   ├── best_yolov8n_kitti.onnx
│   ├── best.onnx
│   ├── twinlitenet.onnx
│   └── midas_small.onnx      # Generated locally; not committed
├── src/
│   ├── detector.py         # ONNX detection and ByteTrack
│   ├── depth.py            # MiDaS depth and distance estimation
│   └── lane.py             # TwinLiteNet/OpenCV lane detection
└── output/                 # Generated annotated videos
```

## Requirements

- Python 3.10 or newer
- A webcam or video file
- Windows, Linux, or macOS
- CPU with enough memory for the selected models

The default configuration is designed for CPU execution. A GPU is not required.

## Installation

From this directory, create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents script activation, activate the environment from a
Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

## MiDaS model export

The `midas_small.onnx` file is a large model artifact and is not intended to be
published on GitHub. ONNX model files are ignored by this repository's
`.gitignore`, so a fresh clone may not contain the required depth model.

Generate the depth model locally with:

```powershell
python export_midas.py
```

This command downloads the pretrained **MiDaS_small** weights through PyTorch
Hub, exports them at the model's native 256×256 input resolution as:

```text
models/midas_small.onnx
```

and verifies the exported ONNX graph with ONNX's model checker. The export
requires an internet connection the first time it is run and may take several
minutes and several gigabytes of temporary disk space. Keep the generated ONNX
file in `models/`; do not commit it to GitHub.

If `onnx` is not already installed in the environment, install it before
running the export:

```powershell
pip install onnx
```

## Running the pipeline

The sample input and output paths are configured in the `__main__` block of
`main.py`. Place a video in `output/` or update `sample_input` to point to
another video, then run:

```powershell
python main.py
```

The application opens a resizable OpenCV window. Press **Q** to stop processing.
The annotated video is written to the output path configured in `main.py`, for
example:

```text
output/external_perception_result1.mp4
```

By default, high-resolution input is resized to a width of 1280 pixels to reduce
CPU, memory, and video-encoding overhead. Detection and depth inference are
intentionally run at different frame intervals to improve throughput.

## Model files

| File | Purpose |
| --- | --- |
| `best_yolov8n_kitti.onnx` or another FP32 YOLO ONNX file | KITTI object detection |
| `twinlitenet.onnx` | Lane segmentation |
| `midas_small.onnx` | MiDaS monocular relative depth |

The detector automatically searches `models/` for an FP32 ONNX model. The lane
detector uses TwinLiteNet when available and otherwise switches to OpenCV lane
detection. The depth estimator uses `midas_small.onnx` when no explicitly
configured depth model is available.

## Important limitations

- MiDaS produces relative depth. Metric distance is an estimate based on camera
  calibration and assumed real-world object heights.
- The default camera horizontal field of view is 70 degrees; adjust it in
  `main.py` if the camera differs.
- The output is a perception and visualization pipeline, not a certified
  autonomous-driving control system.