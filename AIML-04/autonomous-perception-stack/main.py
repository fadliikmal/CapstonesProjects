import os
import time
import queue
import threading
import cv2
from src.detector import ObjectDetectorTracker
from src.depth import MultiprocessDepthWorker
from src.lane import OpenCvLaneDetector, TwinLiteNetLaneDetector


class AsyncVideoWriter:
    """Offloads MP4 disk encoding to a background thread so display FPS is never throttled."""

    def __init__(self, output_path: str, fps: float, frame_size: tuple):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.q = queue.Queue(maxsize=64)
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.q.full():
            self.q.put(frame.copy())

    def _write_loop(self):
        while self.running or not self.q.empty():
            try:
                frame = self.q.get(timeout=0.1)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()


def run_yolo_bytetrack_midas_cpu(
    input_source: str,
    output_path: str = "output/yolo_bytetrack_dpt_hybrid.mp4",
    detect_every_n_frames: int = 2,
    depth_every_n_frames: int = 10,
    target_width: int = 1280,
    show_depth_pip: bool = True,
    save_video: bool = True,
):
    print("[INFO] Starting Direct ONNX Pipeline (YOLO + ByteTrack + TwinLiteNet + MiDaS)...")

    detector = ObjectDetectorTracker(
        model_dir="models",
        conf_thresh=0.15,
        iou_thresh=0.45,
        num_threads=6,
    )
    if os.path.exists("models/twinlitenet.onnx"):
        lane_detector = TwinLiteNetLaneDetector(
            model_path="models/twinlitenet.onnx",
            num_threads=2,
            inference_interval=3,
        )
        print("[INFO] TwinLiteNet lane segmentation + polynomial fitting enabled (CPU ONNX).")
    else:
        lane_detector = OpenCvLaneDetector(processing_width=640)
        print("[WARN] TwinLiteNet model unavailable; using OpenCV lane fallback.")

    depth_worker = MultiprocessDepthWorker(
        model_path="models/midas_dpt_hybrid.onnx",
        scale_factor=1200.0,
        ema_alpha=0.30,
        num_threads=2,
        horizontal_fov_deg=70.0,
    )

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        depth_worker.stop()
        raise FileNotFoundError(f"Failed to open input video: {input_source}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Downscale 4K/high-res input to 720p (1280px wide) to cut memory & encoding overhead by 9x
    if target_width > 0 and raw_w > target_width:
        scale = target_width / float(raw_w)
        orig_w = target_width
        orig_h = int(round(raw_h * scale))
        need_resize = True
        print(f"[INFO] Downscaling input stream from {raw_w}x{raw_h} to {orig_w}x{orig_h} for real-time CPU throughput.")
    else:
        orig_w, orig_h = raw_w, raw_h
        need_resize = False

    window_name = "Direct ONNX: YOLO + ByteTrack + TwinLiteNet + MiDaS"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    win_w = 1080
    win_h = int(win_w * (orig_h / orig_w))
    cv2.resizeWindow(window_name, win_w, win_h)

    video_writer = AsyncVideoWriter(output_path, fps_video, (orig_w, orig_h)) if save_video else None

    font_scale = max(0.55, orig_h / 1000.0)
    box_thick = max(2, int(orig_h / 450.0))
    text_thick = max(1, int(font_scale * 1.8))

    hud_scale = max(0.85, orig_h / 760.0)
    hud_thick = max(2, int(round(hud_scale * 1.8)))

    class_heights = {
        "CAR": 1.50,
        "VAN": 1.90,
        "TRUCK": 3.20,
        "BUS": 3.20,
        "PEDESTRIAN": 1.70,
        "PERSON_SITTING": 1.10,
        "CYCLIST": 1.70,
        "TRAM": 3.20,
    }

    prev_time = time.perf_counter()
    fps_smooth = 0.0
    frame_idx = 0
    last_infer_ms = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if need_resize:
                frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if frame_idx % depth_every_n_frames == 0:
                depth_worker.submit_frame(frame)

            if frame_idx % detect_every_n_frames == 0:
                t_start = time.perf_counter()
                tracked_objects = detector.process(frame)
                last_infer_ms = (time.perf_counter() - t_start) * 1000.0
            else:
                tracked_objects = detector.predict_skipped_frame(orig_w, orig_h)

            # Extract structured LaneState and render ego-lane overlay in-place
            frame, lane_state = lane_detector.process(frame)

            # First pass: compute distance and check ego-lane membership for ACC/AEB lead selection
            enriched_objects = []
            lead_obj = None
            lead_dist = float("inf")

            for obj in tracked_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cls_name = obj["cls_name"].upper()
                track_id = obj["track_id"]

                dist_m = depth_worker.estimate_distance(
                    bbox=(x1, y1, x2, y2),
                    orig_size=(orig_w, orig_h),
                    track_id=track_id,
                    object_height_m=class_heights.get(cls_name),
                )

                foot_x = 0.5 * (x1 + x2)
                foot_y = float(y2)
                in_lane = lane_state.contains_point(foot_x, foot_y)

                item = {
                    "bbox": (x1, y1, x2, y2),
                    "cls_name": cls_name,
                    "track_id": track_id,
                    "dist_m": dist_m,
                    "in_lane": in_lane,
                }
                enriched_objects.append(item)

                if in_lane and 0.0 < dist_m < lead_dist:
                    lead_dist = dist_m
                    lead_obj = item

            # Second pass: render bounding boxes with in-lane & lead vehicle priority
            for item in enriched_objects:
                x1, y1, x2, y2 = item["bbox"]
                cls_name = item["cls_name"]
                track_id = item["track_id"]
                dist_m = item["dist_m"]
                in_lane = item["in_lane"]
                is_lead = lead_obj is not None and item["track_id"] == lead_obj["track_id"]

                dist_str = f" | {dist_m:.1f}m" if dist_m > 0.0 else ""
                role_tag = " [LEAD]" if is_lead else (" [IN-LANE]" if in_lane else "")
                label_text = f"ID:{track_id} {cls_name}{dist_str}{role_tag}"

                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick
                )

                if is_lead and 0.0 < dist_m < 12.0:
                    color = (0, 60, 255)      # High-priority braking alert
                elif is_lead or in_lane:
                    color = (0, 230, 110)     # Active ego-lane target
                elif cls_name in ("TRUCK", "VAN"):
                    color = (0, 180, 255)     # Adjacent lane heavy vehicle
                else:
                    color = (255, 220, 0)     # Adjacent lane vehicle

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
                bg_y1 = max(0, y1 - text_h - 10)
                cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 5, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    text_thick,
                    cv2.LINE_AA,
                )

            if show_depth_pip:
                pip_w = max(200, int(orig_w * 0.22))
                pip_h = int(pip_w * (orig_h / orig_w))
                depth_pip = depth_worker.get_mini_depth_colormap(pip_w, pip_h)
                if depth_pip is not None:
                    px1 = orig_w - pip_w - 18
                    py1 = orig_h - pip_h - 18
                    frame[py1 : py1 + pip_h, px1 : px1 + pip_w] = depth_pip
                    cv2.rectangle(frame, (px1, py1), (px1 + pip_w, py1 + pip_h), (255, 255, 255), 2)

            curr_time = time.perf_counter()
            loop_time = curr_time - prev_time
            prev_time = curr_time

            inst_fps = 1.0 / loop_time if loop_time > 0 else 0.0
            fps_smooth = (0.85 * fps_smooth) + (0.15 * inst_fps) if fps_smooth > 0 else inst_fps

            line1 = f"FPS: {fps_smooth:.1f} | YOLO+BT: {last_infer_ms:.1f}ms | MiDaS: DPT-Hybrid"
            if lane_state.detected:
                ldw_tag = "LDW ALERT" if lane_state.departure_warning else "CENTERED"
                lead_tag = (
                    f"ID:{lead_obj['track_id']} ({lead_dist:.1f}m)"
                    if lead_obj is not None and lead_dist < float("inf")
                    else "CLEAR"
                )
                line2 = (
                    f"Lane ({lane_state.source}): {ldw_tag} | "
                    f"Offset: {lane_state.lateral_offset_m:+.2f}m | Lead: {lead_tag}"
                )
            else:
                line2 = f"Lane ({lane_state.source}): SEARCHING..."

            (w1, h1), base1 = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_thick)
            (w2, h2), base2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, hud_scale * 0.88, hud_thick)

            pad_x, pad_y, line_gap = 18, 14, 12
            box_x1, box_y1 = 15, 15
            box_x2 = box_x1 + max(w1, w2) + (pad_x * 2)
            box_y2 = box_y1 + h1 + base1 + line_gap + h2 + base2 + (pad_y * 2)

            hud_border = (0, 80, 255) if lane_state.departure_warning else (0, 255, 0)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (10, 10, 10), -1)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), hud_border, 3)

            y_text1 = box_y1 + pad_y + h1
            y_text2 = y_text1 + base1 + line_gap + h2
            cv2.putText(
                frame,
                line1,
                (box_x1 + pad_x, y_text1),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale,
                (0, 255, 0),
                hud_thick,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line2,
                (box_x1 + pad_x, y_text2),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale * 0.88,
                (0, 120, 255) if lane_state.departure_warning else (255, 240, 180),
                hud_thick,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        depth_worker.stop()
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Completed. Average smoothed FPS: {fps_smooth:.1f}")


if __name__ == "__main__":
    # sample_input = "output/4608285-uhd_3840_2160_24fps.mp4"
    sample_input = "output/5382495-uhd_3840_2160_24fps.mp4" 
    # sample_input = "output/4608274-uhd_3840_2160_24fps.mp4" 
    # sample_input = "output/12838618_3840_2160_25fps.mp4" 
    if os.path.exists(sample_input):
        run_yolo_bytetrack_midas_cpu(
            input_source=sample_input,
            detect_every_n_frames=2,
            depth_every_n_frames=10,
            target_width=1280,
            show_depth_pip=False,
            output_path="output/external_perception_result1.mp4",
        )
    else:
        print(f"[NOTE] Video file not found at '{sample_input}'. Update 'sample_input' path in main.py.")