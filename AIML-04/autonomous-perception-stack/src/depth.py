import os
import queue
import multiprocessing as mp
from multiprocessing.synchronize import Event
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque


class OnnxDepthEstimator:
    """
    Direct ONNX Runtime CPU Depth Estimator for Intel MiDaS (DPT_Hybrid / MiDaS_small).
    Uses robust relative-depth sampling and optional camera/object calibration.
    """

    def __init__(
        self,
        model_path: str | None = None,
        scale_factor: float | None = None,
        ema_alpha: float = 0.30,
        num_threads: int = 2,
        horizontal_fov_deg: float = 70.0,
    ):
        if not 30.0 <= horizontal_fov_deg <= 150.0:
            raise ValueError("horizontal_fov_deg must be between 30 and 150 degrees")

        self.model_path = self._resolve_model_path(model_path)
        self.ema_alpha = ema_alpha
        self.distance_history: dict[int, float] = {}
        self.distance_samples: dict[int, deque[float]] = {}
        self.horizontal_fov_deg = horizontal_fov_deg

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_h = int(input_meta.shape[2])
        self.input_w = int(input_meta.shape[3])
        self.output_name = self.session.get_outputs()[0].name

        # Match official MiDaS transforms and horizon-subtracted K calibration
        if self.input_h >= 384:
            self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.scale_factor = scale_factor if scale_factor is not None else 1200.0
        else:
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            self.scale_factor = scale_factor if scale_factor is not None else 1100.0

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        if model_path is not None and os.path.exists(model_path):
            return model_path

        dpt_path = "models/midas_dpt_hybrid.onnx"
        small_path = "models/midas_small.onnx"
        if os.path.exists(dpt_path):
            return dpt_path
        if os.path.exists(small_path):
            return small_path

        raise FileNotFoundError(
            "No MiDaS ONNX model found in 'models/'. Run 'python export_midas.py' first."
        )

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[0] != self.input_h or img_bgr.shape[1] != self.input_w:
            img_bgr = cv2.resize(
                img_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - self.mean) / self.std
        blob = np.transpose(img_norm, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def get_depth_map(self, img_bgr: np.ndarray) -> np.ndarray:
        """Returns a native-resolution float32 inverse relative depth map."""
        blob = self._preprocess(img_bgr)
        raw_depth = self.session.run([self.output_name], {self.input_name: blob})[0]
        return np.squeeze(raw_depth).astype(np.float32)

    def get_bbox_distance(
        self,
        depth_map: np.ndarray,
        bbox: tuple,
        orig_size: tuple,
        track_id: int | None = None,
        horizon_baseline: float | None = None,
        object_height_m: float | None = None,
    ) -> float:
        """
        Computes distance from a robust MiDaS sample and, when an object height
        is known, a camera-calibrated pinhole estimate. MiDaS is relative depth,
        so the geometric estimate is the metric source for known object classes.
        """
        orig_w, orig_h = orig_size
        d_h, d_w = depth_map.shape
        object_pixel_height = max(1.0, float(bbox[3] - bbox[1]))

        scale_x = d_w / float(orig_w)
        scale_y = d_h / float(orig_h)

        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        # Use the central lower part of the box: it avoids edges while retaining
        # the object's contact region, which is more stable than the full box.
        bw = x2 - x1
        bh = y2 - y1
        if bw >= 6:
            margin_x = int(bw * 0.25)
            x1 += margin_x
            x2 -= margin_x
        if bh >= 6:
            y1 += int(bh * 0.10)
            y2 -= int(bh * 0.05)

        x1 = max(0, min(d_w - 1, x1))
        y1 = max(0, min(d_h - 1, y1))
        x2 = max(x1 + 1, min(d_w, x2))
        y2 = max(y1 + 1, min(d_h, y2))

        box_roi = depth_map[y1:y2, x1:x2]
        box_roi = box_roi[np.isfinite(box_roi)]
        if box_roi.size == 0:
            return self.distance_history.get(track_id, 0.0) if track_id is not None else 0.0

        if horizon_baseline is None:
            valid_depth = depth_map[np.isfinite(depth_map)]
            horizon_baseline = float(np.percentile(valid_depth, 5))

        # A trimmed median is less sensitive to a few background pixels than
        # the old full-box median.
        median_inv_depth = float(np.percentile(box_roi, 50))
        net_inv_depth = max(1e-3, median_inv_depth - horizon_baseline)
        midas_dist = float(np.clip(self.scale_factor / net_inv_depth, 2.0, 80.0))

        if object_height_m is not None and object_height_m > 0.0 and object_pixel_height > 1:
            # focal/original-image relation from the configured horizontal FOV
            # makes this independent of the MiDaS input resolution.
            focal_px = orig_w / (
                2.0 * np.tan(np.deg2rad(self.horizontal_fov_deg) / 2.0)
            )
            geometric_dist = focal_px * object_height_m / object_pixel_height
            # MiDaS is useful as a sanity check, but should not override the
            # metric estimate for classes with a known physical height.
            raw_dist = float(
                np.clip(0.80 * geometric_dist + 0.20 * midas_dist, 2.0, 80.0)
            )
        else:
            raw_dist = midas_dist

        if track_id is not None:
            samples = self.distance_samples.setdefault(track_id, deque(maxlen=5))
            samples.append(raw_dist)
            robust_dist = float(np.median(np.asarray(samples, dtype=np.float32)))
            if track_id in self.distance_history:
                smoothed = (
                    self.ema_alpha * robust_dist
                    + (1.0 - self.ema_alpha) * self.distance_history[track_id]
                )
            else:
                smoothed = robust_dist
            self.distance_history[track_id] = smoothed
            return smoothed

        return raw_dist


def _depth_worker_loop(
    model_path: str,
    num_threads: int,
    input_q: mp.Queue,
    output_q: mp.Queue,
    stop_event: Event,
):
    """Isolated OS process running MiDaS ONNX completely free from Python GIL."""
    estimator = OnnxDepthEstimator(model_path=model_path, num_threads=num_threads)

    while not stop_event.is_set():
        try:
            resized_bgr = input_q.get(timeout=0.05)
        except queue.Empty:
            continue

        depth_map = estimator.get_depth_map(resized_bgr)

        while not output_q.empty():
            try:
                output_q.get_nowait()
            except queue.Empty:
                break

        try:
            output_q.put_nowait(depth_map)
        except queue.Full:
            pass


class MultiprocessDepthWorker:
    """Zero-blocking multiprocessing wrapper for MiDaS DPT_Hybrid / MiDaS_small ONNX."""

    def __init__(
        self,
        model_path: str | None = None,
        scale_factor: float | None = None,
        ema_alpha: float = 0.30,
        num_threads: int = 2,
        horizontal_fov_deg: float = 70.0,
    ):
        self.helper = OnnxDepthEstimator(
            model_path=model_path,
            scale_factor=scale_factor,
            ema_alpha=ema_alpha,
            num_threads=1,
            horizontal_fov_deg=horizontal_fov_deg,
        )
        self.resolved_path = self.helper.model_path
        self.input_size = (self.helper.input_w, self.helper.input_h)
        self.input_q: mp.Queue = mp.Queue(maxsize=1)
        self.output_q: mp.Queue = mp.Queue(maxsize=1)
        self.stop_event: Event = mp.Event()
        self.latest_depth_map: np.ndarray | None = None
        self.latest_horizon_baseline: float | None = None

        self.process = mp.Process(
            target=_depth_worker_loop,
            args=(self.resolved_path, num_threads, self.input_q, self.output_q, self.stop_event),
            daemon=True,
        )
        self.process.start()
        print(
            f"[INFO] MiDaS ONNX Worker started: {os.path.basename(self.resolved_path)} "
            f"| Grid: {self.input_size[0]}x{self.input_size[1]} | Scale K: {self.helper.scale_factor} "
            f"| Cores: {num_threads}"
        )

    def submit_frame(self, frame_bgr: np.ndarray):
        """Resizes with INTER_LINEAR to preserve distant object depth cues before IPC transfer."""
        if not self.input_q.full():
            resized_bgr = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
            try:
                self.input_q.put_nowait(resized_bgr)
            except queue.Full:
                pass

    def get_latest_depth_map(self) -> np.ndarray | None:
        updated = False
        while not self.output_q.empty():
            try:
                self.latest_depth_map = self.output_q.get_nowait()
                updated = True
            except queue.Empty:
                break
        if updated and self.latest_depth_map is not None:
            valid_depth = self.latest_depth_map[np.isfinite(self.latest_depth_map)]
            self.latest_horizon_baseline = float(np.percentile(valid_depth, 5))
        return self.latest_depth_map

    def estimate_distance(
        self,
        bbox: tuple,
        orig_size: tuple,
        track_id: int | None = None,
        object_height_m: float | None = None,
    ) -> float:
        depth_map = self.get_latest_depth_map()
        if depth_map is None:
            return self.helper.distance_history.get(track_id, 0.0) if track_id is not None else 0.0
        return self.helper.get_bbox_distance(
            depth_map,
            bbox,
            orig_size,
            track_id=track_id,
            horizon_baseline=self.latest_horizon_baseline,
            object_height_m=object_height_m,
        )

    def get_mini_depth_colormap(self, width: int = 240, height: int = 135) -> np.ndarray | None:
        """Renders a compact Inferno depth map preview for real-time visual verification."""
        depth_map = self.get_latest_depth_map()
        if depth_map is None:
            return None
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        norm = ((depth_map - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        return cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)

    def stop(self):
        self.stop_event.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()