import os
import ast
import glob
import cv2
import numpy as np
import onnxruntime as ort


class KalmanBoxTracker:
    """
    Constant-velocity Kalman Filter for 2D bounding box state estimation.
    State vector: [cx, cy, area, aspect_ratio, vx, vy, v_area]
    """

    def __init__(self, bbox: np.ndarray, cls_id: int, conf: float, track_id: int):
        self.track_id = track_id
        self.cls_id = cls_id
        self.conf = conf
        self.time_since_update = 0
        self.hits = 1

        # 7D state vector and 4D measurement vector
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = self._bbox_to_z(bbox)

        # State transition matrix (dt = 1 frame)
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrices
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        self.Q = np.eye(7, dtype=np.float32) * 0.05
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.R[2:, 2:] *= 10.0

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + 0.5 * w
        cy = float(bbox[1]) + 0.5 * h
        area = w * h
        ratio = w / h
        return np.array([[cx], [cy], [area], [ratio]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        cx, cy, area, ratio = float(z[0, 0]), float(z[1, 0]), float(z[2, 0]), float(z[3, 0])
        area = max(1.0, area)
        ratio = max(0.05, ratio)
        w = np.sqrt(area * ratio)
        h = area / w
        return np.array(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            dtype=np.float32,
        )

    def predict(self) -> np.ndarray:
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox: np.ndarray, cls_id: int, conf: float):
        self.time_since_update = 0
        self.hits += 1
        self.cls_id = cls_id
        self.conf = conf

        z = self._bbox_to_z(bbox)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        identity = np.eye(7, dtype=np.float32)
        self.P = (identity - (K @ self.H)) @ self.P

    def get_bbox(self) -> np.ndarray:
        return self._z_to_bbox(self.x[:4])


class StandaloneByteTracker:
    """
    Lightweight Pure-NumPy ByteTrack implementation:
    - Stage 1: Associates high-confidence detections with active Kalman tracks.
    - Stage 2: Recovers occluded objects using low-confidence detections.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.40,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.45,
        match_thresh_high: float = 0.25,
        match_thresh_low: float = 0.40,
        max_age: int = 20,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_age = max_age

        self.next_id = 1
        self.tracks: list[KalmanBoxTracker] = []

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
        area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

        ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter_area = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union_area = area_a[:, None] + area_b[None, :] - inter_area + 1e-6
        return inter_area / union_area

    def _associate(self, track_indices: list, det_boxes: np.ndarray, iou_thresh: float):
        if len(track_indices) == 0 or len(det_boxes) == 0:
            return [], list(track_indices), list(range(len(det_boxes)))

        trk_boxes = np.array([self.tracks[i].get_bbox() for i in track_indices], dtype=np.float32)
        iou_mat = self._compute_iou_matrix(trk_boxes, det_boxes)

        matches = []
        matched_t = set()
        matched_d = set()

        while True:
            t_local, d_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[t_local, d_idx] < iou_thresh:
                break
            matches.append((track_indices[t_local], int(d_idx)))
            matched_t.add(t_local)
            matched_d.add(int(d_idx))
            iou_mat[t_local, :] = -1.0
            iou_mat[:, d_idx] = -1.0

        unmatched_tracks = [track_indices[i] for i in range(len(track_indices)) if i not in matched_t]
        unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def predict_only(self) -> list:
        """Advances Kalman states on skipped frames without running neural network inference."""
        outputs = []
        alive_tracks = []
        for trk in self.tracks:
            pred_box = trk.predict()
            if trk.time_since_update <= self.max_age:
                alive_tracks.append(trk)
                if trk.time_since_update <= 3:
                    outputs.append((pred_box, trk.conf, trk.cls_id, trk.track_id))
        self.tracks = alive_tracks
        return outputs

    def update(self, detections: list) -> list:
        """
        Args:
            detections: list of tuples (x1, y1, x2, y2, conf, cls_id)
        Returns:
            list of tuples (bbox_ndarray, conf, cls_id, track_id)
        """
        # 1. Predict next state for all existing tracks
        for trk in self.tracks:
            trk.predict()

        # 2. Split detections into high-confidence and low-confidence pools
        high_dets = [d for d in detections if d[4] >= self.track_high_thresh]
        low_dets = [d for d in detections if self.track_low_thresh <= d[4] < self.track_high_thresh]

        all_track_indices = list(range(len(self.tracks)))

        # 3. Stage 1: Match active tracks with high-confidence detections
        high_boxes = np.array([d[:4] for d in high_dets], dtype=np.float32) if high_dets else np.empty((0, 4), dtype=np.float32)
        matches_1, unmatched_tracks_1, unmatched_high_dets = self._associate(
            all_track_indices, high_boxes, self.match_thresh_high
        )

        for t_idx, d_idx in matches_1:
            det = high_dets[d_idx]
            self.tracks[t_idx].update(np.array(det[:4], dtype=np.float32), det[5], det[4])

        # 4. Stage 2: Match remaining recently-seen tracks with low-confidence detections
        recent_unmatched = [i for i in unmatched_tracks_1 if self.tracks[i].time_since_update <= 3]
        low_boxes = np.array([d[:4] for d in low_dets], dtype=np.float32) if low_dets else np.empty((0, 4), dtype=np.float32)
        matches_2, _, _ = self._associate(
            recent_unmatched, low_boxes, self.match_thresh_low
        )

        for t_idx, d_idx in matches_2:
            det = low_dets[d_idx]
            self.tracks[t_idx].update(np.array(det[:4], dtype=np.float32), det[5], det[4])

        # 5. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            if det[4] >= self.new_track_thresh:
                new_trk = KalmanBoxTracker(
                    bbox=np.array(det[:4], dtype=np.float32),
                    cls_id=det[5],
                    conf=det[4],
                    track_id=self.next_id,
                )
                self.next_id += 1
                self.tracks.append(new_trk)

        # 6. Prune expired tracks and collect active outputs
        outputs = []
        alive_tracks = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                alive_tracks.append(trk)
                if trk.time_since_update == 0:
                    outputs.append((trk.get_bbox(), trk.conf, trk.cls_id, trk.track_id))

        self.tracks = alive_tracks
        return outputs


class ObjectDetectorTracker:
    """Direct ONNX Runtime CPU inference engine coupled with Standalone ByteTrack."""

    DEFAULT_KITTI_NAMES = {
        0: "car",
        1: "van",
        2: "truck",
        3: "pedestrian",
        4: "Person_sitting",
        5: "cyclist",
        6: "tram",
        7: "misc",
        8: "DontCare",
    }

    def __init__(
        self,
        model_dir: str = "models",
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        num_threads: int = 8,
    ):
        # Keep low_thresh at 0.15 so ByteTrack Stage-2 can recover occluded objects
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_path = self._resolve_onnx_model(model_dir)

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

        meta_map = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta_map:
            self.names = ast.literal_eval(meta_map["names"])
        else:
            self.names = self.DEFAULT_KITTI_NAMES

        self.tracker = StandaloneByteTracker(
            track_high_thresh=0.38,
            track_low_thresh=0.15,
            new_track_thresh=0.42,
            max_age=20,
        )
        print(
            f"[INFO] Direct ONNX + ByteTrack ready: {os.path.basename(self.model_path)} "
            f"| Input: {self.input_w}x{self.input_h} | CPU Cores: {num_threads}"
        )

    @staticmethod
    def _resolve_onnx_model(model_dir: str) -> str:
        preferred = os.path.join(model_dir, "best_yolov8n_kitti_384x640_fp32.onnx")
        if os.path.exists(preferred):
            return preferred

        fp32_candidates = sorted(glob.glob(os.path.join(model_dir, "*fp32*.onnx")))
        if fp32_candidates:
            return fp32_candidates[0]

        any_onnx = [
            p for p in sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
            if "int8" not in os.path.basename(p).lower()
        ]
        if any_onnx:
            return any_onnx[0]

        raise FileNotFoundError(
            f"No FP32 ONNX model found in '{model_dir}'. "
            "Place 'best_yolov8n_kitti_384x640_fp32.onnx' inside 'models/'."
        )

    def _format_results(self, tracked_items: list, orig_w: int, orig_h: int) -> list:
        results = []
        for box_arr, conf, cls_id, track_id in tracked_items:
            x1 = max(0, int(box_arr[0]))
            y1 = max(0, int(box_arr[1]))
            x2 = min(orig_w, int(box_arr[2]))
            y2 = min(orig_h, int(box_arr[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf),
                    "cls_id": int(cls_id),
                    "cls_name": self.names.get(int(cls_id), f"cls_{cls_id}"),
                    "track_id": int(track_id),
                }
            )
        return results

    def predict_skipped_frame(self, orig_w: int, orig_h: int) -> list:
        """Extrapolates bounding boxes via Kalman Filter on skipped frames for high FPS."""
        tracked = self.tracker.predict_only()
        return self._format_results(tracked, orig_w, orig_h)

    def process(self, frame: np.ndarray) -> list:
        """Runs full ONNX inference + C++ NMS + ByteTrack association."""
        orig_h, orig_w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_w, self.input_h),
            swapRB=True,
            crop=False,
        )

        raw_output = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        if raw_output.shape[0] < raw_output.shape[1]:
            raw_output = raw_output.T

        class_scores = raw_output[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Filter out low scores and KITTI 'DontCare' class (ID 8)
        valid_mask = (confidences >= self.conf_thresh) & (class_ids != 8)
        if not np.any(valid_mask):
            tracked = self.tracker.update([])
            return self._format_results(tracked, orig_w, orig_h)

        boxes_cxcywh = raw_output[valid_mask, :4]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]

        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        cx = boxes_cxcywh[:, 0] * scale_x
        cy = boxes_cxcywh[:, 1] * scale_y
        w = boxes_cxcywh[:, 2] * scale_x
        h = boxes_cxcywh[:, 3] * scale_y

        x1 = (cx - 0.5 * w).astype(int)
        y1 = (cy - 0.5 * h).astype(int)
        w_int = w.astype(int)
        h_int = h.astype(int)

        nms_boxes = np.column_stack((x1, y1, w_int, h_int)).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self.conf_thresh, self.iou_thresh)

        detections = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx1 = max(0, x1[idx])
                by1 = max(0, y1[idx])
                bx2 = min(orig_w, x1[idx] + w_int[idx])
                by2 = min(orig_h, y1[idx] + h_int[idx])
                detections.append((bx1, by1, bx2, by2, float(confidences[idx]), int(class_ids[idx])))

        tracked = self.tracker.update(detections)
        return self._format_results(tracked, orig_w, orig_h)