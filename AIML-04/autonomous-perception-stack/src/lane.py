from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class LaneState:
    """Structured lane perception state for downstream planning, control, and sensor fusion."""

    detected: bool
    confidence: float
    left_poly_norm: np.ndarray | None       # [a, b, c] for x_norm = a*y_norm^2 + b*y_norm + c
    right_poly_norm: np.ndarray | None      # [a, b, c]
    center_poly_norm: np.ndarray | None     # [a, b, c]
    polygon_px: np.ndarray | None           # (2N, 2) int32 ego-lane corridor polygon
    left_points_px: np.ndarray | None       # (N, 2) int32
    right_points_px: np.ndarray | None      # (N, 2) int32
    center_points_px: np.ndarray | None     # (N, 2) int32
    lateral_offset_ratio: float             # -1.0 (left of lane center) to +1.0 (right)
    lateral_offset_m: float                 # Estimated offset in meters (3.7m standard lane)
    heading_error_deg: float                # Road centerline angle deviation in degrees
    departure_warning: bool                 # True when vehicle drifts near lane boundary
    frame_size: tuple[int, int]             # (width, height)
    source: str                             # "TwinLiteNet" or "OpenCV"

    def contains_point(self, x: float, y: float, margin_ratio: float = 0.05) -> bool:
        """
        Checks whether an image coordinate (e.g., vehicle bottom-center contact point)
        lies inside the detected ego-lane corridor.
        """
        if not self.detected or self.left_poly_norm is None or self.right_poly_norm is None:
            return False

        width, height = self.frame_size
        if width <= 0 or height <= 0:
            return False

        y_norm = y / float(height)
        x_norm = x / float(width)

        if not (0.40 <= y_norm <= 1.03):
            return False

        left_x = float(np.polyval(self.left_poly_norm, y_norm))
        right_x = float(np.polyval(self.right_poly_norm, y_norm))
        lane_width = right_x - left_x
        if lane_width <= 0.02:
            return False

        margin = lane_width * margin_ratio
        return (left_x - margin) <= x_norm <= (right_x + margin)


class _BaseLaneGeometryMixin:
    """Shared geometry validation, low-light enhancement, telemetry, and ROI-cropped renderer."""

    STANDARD_LANE_WIDTH_M = 3.7
    DEFAULT_WIDTH_POLY = np.array([0.0, 0.62, -0.24], dtype=np.float32)
    _CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    @classmethod
    def _prepare_lighting(cls, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """Applies fast CLAHE luminance equalization automatically on dark/night frames."""
        mean_luma = float(np.mean(image[::8, ::8]))
        if mean_luma >= 85.0:
            return image, False

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_eq = cls._CLAHE.apply(l_chan)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
        return enhanced, True

    @staticmethod
    def _valid_norm_pair(left_poly: np.ndarray, right_poly: np.ndarray) -> bool:
        """Validates perspective geometry across both wide-FOV and narrow-FOV cameras."""
        y_top = 0.50
        y_mid = 0.73
        y_bot = 0.95

        l_top, l_mid, l_bot = np.polyval(left_poly, [y_top, y_mid, y_bot])
        r_top, r_mid, r_bot = np.polyval(right_poly, [y_top, y_mid, y_bot])

        w_top = r_top - l_top
        w_mid = r_mid - l_mid
        w_bot = r_bot - l_bot

        return bool(
            np.isfinite([l_top, r_top, l_mid, r_mid, l_bot, r_bot]).all()
            and l_top < r_top
            and l_mid < r_mid
            and l_bot < r_bot
            and 0.02 <= w_top <= 0.65
            and 0.12 <= w_bot <= 0.98
            and w_bot > w_top * 1.08
            and w_mid > w_top * 1.02
        )

    def _build_lane_state(
        self,
        left_poly: np.ndarray | None,
        right_poly: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        confidence: float,
        source: str,
        y_top_ratio: float = 0.48,
    ) -> LaneState:
        """Constructs a complete LaneState with pixel polylines and lateral control metrics."""
        if left_poly is None or right_poly is None or confidence <= 0.0:
            return LaneState(
                detected=False,
                confidence=0.0,
                left_poly_norm=None,
                right_poly_norm=None,
                center_poly_norm=None,
                polygon_px=None,
                left_points_px=None,
                right_points_px=None,
                center_points_px=None,
                lateral_offset_ratio=0.0,
                lateral_offset_m=0.0,
                heading_error_deg=0.0,
                departure_warning=False,
                frame_size=(orig_w, orig_h),
                source=source,
            )

        center_poly = 0.5 * (left_poly + right_poly)

        y_eval = 0.92
        left_x_eval = float(np.polyval(left_poly, y_eval))
        right_x_eval = float(np.polyval(right_poly, y_eval))
        lane_center_x = 0.5 * (left_x_eval + right_x_eval)
        half_lane_w = max(1e-4, 0.5 * (right_x_eval - left_x_eval))

        camera_center_x = 0.50
        offset_ratio = float(np.clip((camera_center_x - lane_center_x) / half_lane_w, -1.5, 1.5))
        offset_m = offset_ratio * (self.STANDARD_LANE_WIDTH_M * 0.5)

        x_near = float(np.polyval(center_poly, 0.90)) * orig_w
        x_far = float(np.polyval(center_poly, 0.58)) * orig_w
        dy_px = (0.90 - 0.58) * orig_h
        heading_deg = float(np.degrees(np.arctan2(x_far - x_near, max(1.0, dy_px))))

        plot_y_norm = np.linspace(y_top_ratio, 0.99, 32)
        plot_y_px = plot_y_norm * orig_h

        left_x_px = np.clip(np.polyval(left_poly, plot_y_norm) * orig_w, 0, orig_w - 1)
        right_x_px = np.clip(np.polyval(right_poly, plot_y_norm) * orig_w, 0, orig_w - 1)
        center_x_px = np.clip(np.polyval(center_poly, plot_y_norm) * orig_w, 0, orig_w - 1)

        left_pts = np.column_stack((left_x_px, plot_y_px)).astype(np.int32)
        right_pts = np.column_stack((right_x_px, plot_y_px)).astype(np.int32)
        center_pts = np.column_stack((center_x_px, plot_y_px)).astype(np.int32)
        polygon_pts = np.vstack((left_pts, right_pts[::-1]))

        return LaneState(
            detected=True,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            left_poly_norm=left_poly.astype(np.float32),
            right_poly_norm=right_poly.astype(np.float32),
            center_poly_norm=center_poly.astype(np.float32),
            polygon_px=polygon_pts,
            left_points_px=left_pts,
            right_points_px=right_pts,
            center_points_px=center_pts,
            lateral_offset_ratio=offset_ratio,
            lateral_offset_m=offset_m,
            heading_error_deg=heading_deg,
            departure_warning=abs(offset_ratio) > 0.38,
            frame_size=(orig_w, orig_h),
            source=source,
        )

    @staticmethod
    def draw_overlay(frame: np.ndarray, state: LaneState) -> np.ndarray:
        """Renders ego-lane corridor, boundary curves, and centerline in-place."""
        if (
            not state.detected
            or state.polygon_px is None
            or state.left_points_px is None
            or state.right_points_px is None
        ):
            return frame

        orig_h, orig_w = frame.shape[:2]
        line_thick = max(2, int(round(orig_h / 320.0)))
        pad = line_thick * 3

        all_pts = state.polygon_px
        rx1 = max(0, int(np.min(all_pts[:, 0])) - pad)
        ry1 = max(0, int(np.min(all_pts[:, 1])) - pad)
        rx2 = min(orig_w, int(np.max(all_pts[:, 0])) + pad)
        ry2 = min(orig_h, int(np.max(all_pts[:, 1])) + pad)

        if rx2 <= rx1 or ry2 <= ry1:
            return frame

        offset = np.array([rx1, ry1], dtype=np.int32)
        sub_frame = frame[ry1:ry2, rx1:rx2]
        sub_overlay = sub_frame.copy()

        poly_local = state.polygon_px - offset
        left_local = state.left_points_px - offset
        right_local = state.right_points_px - offset

        fill_color = (0, 70, 230) if state.departure_warning else (0, 180, 60)
        edge_color = (0, 120, 255) if state.departure_warning else (0, 255, 110)

        cv2.fillPoly(sub_overlay, [poly_local], fill_color)
        cv2.polylines(sub_overlay, [left_local], False, edge_color, line_thick, cv2.LINE_AA)
        cv2.polylines(sub_overlay, [right_local], False, edge_color, line_thick, cv2.LINE_AA)

        if state.center_points_px is not None and len(state.center_points_px) >= 4:
            center_local = state.center_points_px - offset
            center_thick = max(1, line_thick // 2)
            for idx in range(0, len(center_local) - 1, 2):
                p1 = tuple(center_local[idx])
                p2 = tuple(center_local[idx + 1])
                cv2.line(sub_overlay, p1, p2, (255, 240, 180), center_thick, cv2.LINE_AA)

        alpha = 0.34
        cv2.addWeighted(sub_overlay, alpha, sub_frame, 1.0 - alpha, 0.0, dst=sub_frame)
        return frame


class OpenCvLaneDetector(_BaseLaneGeometryMixin):
    """Day/Night adaptive classical lane detector with HLS paint filtering and polynomial tracking."""

    def __init__(
        self,
        roi_height_ratio: float = 0.42,
        smoothing: float = 0.30,
        max_missed_frames: int = 10,
        processing_width: int = 640,
    ):
        if not 0.25 <= roi_height_ratio < 0.70:
            raise ValueError("roi_height_ratio must be between 0.25 and 0.70")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")
        if processing_width < 320:
            raise ValueError("processing_width must be at least 320 pixels")

        self.roi_height_ratio = roi_height_ratio
        self.smoothing = smoothing
        self.max_missed_frames = max_missed_frames
        self.processing_width = processing_width

        self.prev_left: np.ndarray | None = None
        self.prev_right: np.ndarray | None = None
        self.missed_frames = 0
        self.last_shape: tuple[int, int] | None = None
        self.mask_history: deque[np.ndarray] = deque(maxlen=4)

    def _mask(self, image: np.ndarray, is_night: bool) -> np.ndarray:
        height, width = image.shape[:2]
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hue, lightness, saturation = cv2.split(hls)

        # Lower luminance floor at night so dim markings outside high-beams remain visible
        min_white = 75 if is_night else 130
        white_level = max(min_white, int(np.percentile(lightness, 78 if is_night else 68)))
        white = ((lightness >= white_level) & (saturation <= 105)).astype(np.uint8) * 255
        yellow = (
            (hue >= 10) & (hue <= 45) & (saturation >= 65) & (lightness >= 50)
        ).astype(np.uint8) * 255

        mask = cv2.bitwise_or(white, yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

        roi = np.zeros_like(mask)
        vertices = np.array(
            [
                [
                    (int(width * 0.06), height - 1),
                    (int(width * 0.36), int(height * self.roi_height_ratio)),
                    (int(width * 0.64), int(height * self.roi_height_ratio)),
                    (int(width * 0.94), height - 1),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(roi, vertices, 255)
        return cv2.bitwise_and(mask, roi)

    @staticmethod
    def _line_parameters(line: np.ndarray) -> tuple[float, float, float] | None:
        x1, y1, x2, y2 = [float(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1.0 or abs(dy) < 8.0:
            return None

        image_slope = dy / dx
        if abs(image_slope) < 0.16 or abs(image_slope) > 6.0:
            return None

        depth_slope = dx / dy
        intercept = x1 - depth_slope * y1
        length = float(np.hypot(dx, dy))
        return depth_slope, intercept, length

    def _find_side(
        self,
        lines: np.ndarray | None,
        width: int,
        height: int,
        side: str,
    ) -> np.ndarray | None:
        if lines is None:
            return None

        y_reference = height * 0.78
        target_x = width * (0.35 if side == "left" else 0.65)
        candidates: list[tuple[float, float, float]] = []

        for line in lines.reshape(-1, 4):
            params = self._line_parameters(line)
            if params is None:
                continue
            slope, intercept, length = params
            x_reference = slope * y_reference + intercept

            if side == "left" and x_reference >= width * 0.50:
                continue
            if side == "right" and x_reference <= width * 0.50:
                continue
            if not width * 0.06 <= x_reference <= width * 0.94:
                continue

            position_error = abs(x_reference - target_x) / width
            score = position_error - min(length / width, 0.30) * 0.35
            candidates.append((score, slope, intercept))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, slope_px, intercept_px = candidates[0]

        norm_b = slope_px * (height / float(width))
        norm_c = intercept_px / float(width)
        return np.array([0.0, norm_b, norm_c], dtype=np.float32)

    def _smooth(self, previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current
        return ((1.0 - self.smoothing) * previous + self.smoothing * current).astype(np.float32)

    def analyze(self, image: np.ndarray, pre_enhanced: tuple[np.ndarray, bool] | None = None) -> LaneState:
        """Computes structured LaneState without modifying the input frame."""
        original_height, original_width = image.shape[:2]
        if self.last_shape != (original_height, original_width):
            self.prev_left = None
            self.prev_right = None
            self.missed_frames = 0
            self.mask_history.clear()
            self.last_shape = (original_height, original_width)

        proc_img, is_night = pre_enhanced if pre_enhanced is not None else self._prepare_lighting(image)

        work_image = proc_img
        if original_width > self.processing_width:
            work_height = max(1, int(original_height * self.processing_width / original_width))
            work_image = cv2.resize(
                proc_img,
                (self.processing_width, work_height),
                interpolation=cv2.INTER_AREA,
            )

        height, width = work_image.shape[:2]
        mask = self._mask(work_image, is_night=is_night)
        self.mask_history.append(mask)
        accumulated_mask = self.mask_history[0].copy()
        for historical_mask in list(self.mask_history)[1:]:
            accumulated_mask = cv2.bitwise_or(accumulated_mask, historical_mask)

        lines = cv2.HoughLinesP(
            accumulated_mask,
            1,
            np.pi / 180.0,
            threshold=max(10, width // 48),
            minLineLength=max(12, width // 28),
            maxLineGap=max(35, width // 6),
        )
        left = self._find_side(lines, width, height, "left")
        right = self._find_side(lines, width, height, "right")

        width_poly = (
            (self.prev_right - self.prev_left)
            if (self.prev_left is not None and self.prev_right is not None)
            else self.DEFAULT_WIDTH_POLY
        )
        if left is not None and right is None:
            right = left + width_poly
        elif right is not None and left is None:
            left = right - width_poly

        if left is not None and right is not None and self._valid_norm_pair(left, right):
            self.prev_left = self._smooth(self.prev_left, left)
            self.prev_right = self._smooth(self.prev_right, right)
            self.missed_frames = 0
        else:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.prev_left = None
                self.prev_right = None

        confidence = (
            max(0.0, 1.0 - (self.missed_frames / (self.max_missed_frames + 1.0)))
            if (self.prev_left is not None and self.prev_right is not None)
            else 0.0
        )
        return self._build_lane_state(
            left_poly=self.prev_left,
            right_poly=self.prev_right,
            orig_w=original_width,
            orig_h=original_height,
            confidence=confidence,
            source="OpenCV",
            y_top_ratio=max(0.46, self.roi_height_ratio + 0.04),
        )

    def process(self, image: np.ndarray) -> tuple[np.ndarray, LaneState]:
        """Analyzes lane geometry and renders in-place overlay, returning (frame, LaneState)."""
        state = self.analyze(image)
        self.draw_overlay(image, state)
        return image, state


class TwinLiteNetLaneDetector(_BaseLaneGeometryMixin):
    """
    FOV-adaptive & Night-enhanced ONNX TwinLiteNet detector with automatic
    classical OpenCV fallback when deep segmentation encounters out-of-domain scenes.
    """

    def __init__(
        self,
        model_path: str = "models/twinlitenet.onnx",
        num_threads: int = 2,
        mask_threshold: float = 0.34,
        inference_interval: int = 3,
        smoothing: float = 0.35,
        max_missed_frames: int = 12,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TwinLiteNet model not found: {model_path}. "
                "Use OpenCvLaneDetector or place twinlitenet.onnx in models/."
            )
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if inference_interval < 1:
            raise ValueError("inference_interval must be at least 1")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            options,
            providers=["CPUExecutionProvider"],
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        raw_h = input_meta.shape[2]
        raw_w = input_meta.shape[3]
        self.input_h = int(raw_h) if isinstance(raw_h, (int, np.integer)) and raw_h > 0 else 360
        self.input_w = int(raw_w) if isinstance(raw_w, (int, np.integer)) and raw_w > 0 else 640

        self.output_names = [output.name for output in self.session.get_outputs()]
        if len(self.output_names) < 2:
            raise ValueError("TwinLiteNet must provide drivable-area and lane outputs")

        self.mask_threshold = min(mask_threshold, 0.36)
        self.inference_interval = inference_interval
        self.smoothing = smoothing
        self.max_missed_frames = max(max_missed_frames, 10)

        self.frame_index = 0
        self.prev_left: np.ndarray | None = None
        self.prev_right: np.ndarray | None = None
        self.missed_frames = 0
        self.last_shape: tuple[int, int] | None = None

        # Broadened trapezoid supports wide-angle, narrow-FOV, and varied camera pitch angles
        self.roi_top_ratio = 0.40
        self._focus_mask = np.zeros((self.input_h, self.input_w), dtype=np.uint8)
        focus_vertices = np.array(
            [
                [
                    (int(self.input_w * 0.05), self.input_h - 1),
                    (int(self.input_w * 0.35), int(self.input_h * self.roi_top_ratio)),
                    (int(self.input_w * 0.65), int(self.input_h * self.roi_top_ratio)),
                    (int(self.input_w * 0.95), self.input_h - 1),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(self._focus_mask, focus_vertices, 255)
        self._morph_kernel = np.ones((3, 3), dtype=np.uint8)
        self._da_dilate_kernel = np.ones((25, 25), dtype=np.uint8)

        # Hybrid fallback detector when TwinLiteNet struggles with extreme night glare or FOV
        self._fallback_detector = OpenCvLaneDetector(
            roi_height_ratio=self.roi_top_ratio,
            smoothing=smoothing,
            max_missed_frames=max_missed_frames,
        )

    def _predict_clean_lane_mask(self, image: np.ndarray, is_night: bool) -> np.ndarray:
        """Runs TwinLiteNet ONNX with night-adaptive thresholding."""
        resized = cv2.resize(
            image,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_AREA,
        )
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255.0,
            size=(self.input_w, self.input_h),
            swapRB=True,
            crop=False,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        da_output = np.asarray(outputs[0])[0]
        lane_output = np.asarray(outputs[1])[0]
        if lane_output.ndim != 3 or lane_output.shape[0] < 2:
            raise ValueError(f"Unexpected TwinLiteNet lane output shape: {lane_output.shape}")

        eff_thresh = max(0.24, self.mask_threshold - 0.08) if is_night else self.mask_threshold
        lane_mask = (
            (lane_output[1] > lane_output[0]) & (lane_output[1] >= eff_thresh)
        ).astype(np.uint8) * 255

        # Skip drivable-area clipping at night because headlight glare distorts drivable masks
        if not is_night and da_output.ndim == 3 and da_output.shape[0] >= 2:
            da_mask = (da_output[1] > da_output[0]).astype(np.uint8) * 255
            if np.count_nonzero(da_mask) > int(0.05 * self.input_h * self.input_w):
                da_corridor = cv2.dilate(da_mask, self._da_dilate_kernel, iterations=1)
                lane_mask = cv2.bitwise_and(lane_mask, da_corridor)

        lane_mask = cv2.bitwise_and(lane_mask, self._focus_mask)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, self._morph_kernel)
        return lane_mask

    @staticmethod
    def _fit_normalized_poly(
        ys_px: np.ndarray,
        xs_px: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray | None, float]:
        if len(ys_px) < 6:
            return None, 0.0

        y_span = (float(np.max(ys_px)) - float(np.min(ys_px))) / float(height)
        if y_span < 0.03:
            return None, y_span

        ys_norm = ys_px.astype(np.float64) / float(height)
        xs_norm = xs_px.astype(np.float64) / float(width)

        if y_span >= 0.20 and len(ys_px) >= 24:
            coeffs = np.polyfit(ys_norm, xs_norm, deg=2)
            if abs(coeffs[0]) > 1.2:
                lin = np.polyfit(ys_norm, xs_norm, deg=1)
                coeffs = np.array([0.0, lin[0], lin[1]], dtype=np.float64)
        else:
            lin = np.polyfit(ys_norm, xs_norm, deg=1)
            coeffs = np.array([0.0, lin[0], lin[1]], dtype=np.float64)

        return coeffs.astype(np.float32), y_span

    def _extract_ego_polynomials(
        self,
        lane_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
        y_start = int(self.input_h * (self.roi_top_ratio + 0.03))
        y_end = self.input_h - 2

        left_ys: list[int] = []
        left_xs: list[float] = []
        right_ys: list[int] = []
        right_xs: list[float] = []

        has_prior = self.prev_left is not None and self.prev_right is not None

        for y in range(y_end, y_start, -1):
            row_xs = np.flatnonzero(lane_mask[y])
            if row_xs.size == 0:
                continue

            if has_prior:
                y_norm = y / float(self.input_h)
                center_x = 0.5 * (
                    float(np.polyval(self.prev_left, y_norm))
                    + float(np.polyval(self.prev_right, y_norm))
                ) * self.input_w
            else:
                center_x = 0.5 * self.input_w

            left_candidates = row_xs[row_xs < center_x]
            right_candidates = row_xs[row_xs > center_x]

            if left_candidates.size > 0:
                inner_right_edge = int(left_candidates[-1])
                cluster = left_candidates[left_candidates >= inner_right_edge - 16]
                left_ys.append(y)
                left_xs.append(float(np.median(cluster)))

            if right_candidates.size > 0:
                inner_left_edge = int(right_candidates[0])
                cluster = right_candidates[right_candidates <= inner_left_edge + 16]
                right_ys.append(y)
                right_xs.append(float(np.median(cluster)))

        left_poly, left_span = self._fit_normalized_poly(
            np.asarray(left_ys), np.asarray(left_xs), self.input_w, self.input_h
        )
        right_poly, right_span = self._fit_normalized_poly(
            np.asarray(right_ys), np.asarray(right_xs), self.input_w, self.input_h
        )
        return left_poly, right_poly, left_span, right_span

    def _smooth(self, previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current
        return ((1.0 - self.smoothing) * previous + self.smoothing * current).astype(np.float32)

    def analyze(self, image: np.ndarray) -> LaneState:
        """Runs scheduled TwinLiteNet inference with automatic OpenCV hybrid fallback."""
        original_height, original_width = image.shape[:2]
        if self.last_shape != (original_height, original_width):
            self.prev_left = None
            self.prev_right = None
            self.missed_frames = 0
            self.frame_index = 0
            self.last_shape = (original_height, original_width)

        is_locked = self.prev_left is not None and self.prev_right is not None
        active_interval = self.inference_interval if is_locked else min(2, self.inference_interval)
        run_inference = (self.frame_index % active_interval == 0)
        self.frame_index += 1

        enhanced_pair: tuple[np.ndarray, bool] | None = None

        if run_inference:
            enhanced_pair = self._prepare_lighting(image)
            proc_img, is_night = enhanced_pair
            lane_mask = self._predict_clean_lane_mask(proc_img, is_night=is_night)
            left_poly, right_poly, left_span, right_span = self._extract_ego_polynomials(lane_mask)

            width_poly = (
                (self.prev_right - self.prev_left)
                if (self.prev_left is not None and self.prev_right is not None)
                else self.DEFAULT_WIDTH_POLY
            )

            if left_poly is not None and right_poly is None and left_span >= 0.08:
                right_poly = left_poly + width_poly
            elif right_poly is not None and left_poly is None and right_span >= 0.08:
                left_poly = right_poly - width_poly
            elif (
                left_poly is not None
                and right_poly is not None
                and not self._valid_norm_pair(left_poly, right_poly)
            ):
                if right_span >= 0.12 and right_span > left_span * 1.5:
                    left_poly = right_poly - width_poly
                elif left_span >= 0.12 and left_span > right_span * 1.5:
                    right_poly = left_poly + width_poly

            if (
                left_poly is not None
                and right_poly is not None
                and self._valid_norm_pair(left_poly, right_poly)
            ):
                self.prev_left = self._smooth(self.prev_left, left_poly)
                self.prev_right = self._smooth(self.prev_right, right_poly)
                self.missed_frames = 0
            else:
                self.missed_frames += 1
                if self.missed_frames > self.max_missed_frames:
                    self.prev_left = None
                    self.prev_right = None

        if self.prev_left is not None and self.prev_right is not None:
            confidence = max(0.0, 1.0 - (self.missed_frames / (self.max_missed_frames + 1.0)))
            return self._build_lane_state(
                left_poly=self.prev_left,
                right_poly=self.prev_right,
                orig_w=original_width,
                orig_h=original_height,
                confidence=confidence,
                source="TwinLiteNet",
                y_top_ratio=self.roi_top_ratio + 0.06,
            )

        # Automatic fallback to OpenCV lane tracker when TwinLiteNet cannot lock
        if enhanced_pair is None:
            enhanced_pair = self._prepare_lighting(image)
        return self._fallback_detector.analyze(image, pre_enhanced=enhanced_pair)

    def process(self, image: np.ndarray) -> tuple[np.ndarray, LaneState]:
        """Analyzes lane geometry and renders in-place overlay, returning (frame, LaneState)."""
        state = self.analyze(image)
        self.draw_overlay(image, state)
        return image, state