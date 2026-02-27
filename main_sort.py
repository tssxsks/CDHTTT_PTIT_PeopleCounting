import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from imutils.video import FPS
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
DEFAULT_INPUT_PATH = PROJECT_DIR / "Input" / "Test.mp4"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "Final_output_sort.mp4"


def disable_ultralytics_integration_callbacks():
    """
    Tat callback tich hop cua Ultralytics (TensorBoard, MLflow, ...),
    tranh loi moi truong package khi chi su dung detect/track.
    """
    try:
        from ultralytics.yolo.utils import callbacks as callbacks_pkg
        from ultralytics.yolo.utils.callbacks import base as callbacks_base

        def _noop_add_integration_callbacks(_instance):
            return None

        callbacks_base.add_integration_callbacks = _noop_add_integration_callbacks
        callbacks_pkg.add_integration_callbacks = _noop_add_integration_callbacks
        logger.info("Da tat callback tich hop cua Ultralytics cho che do su dung.")
    except Exception as exc:
        logger.warning("Khong tat duoc callback tich hop cua Ultralytics: %s", exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="He thong dem so nguoi ra vao tu video (YOLOv8 + SORT)"
    )
    parser.add_argument("--source", default=str(DEFAULT_INPUT_PATH), help="Video/camera/stream dau vao.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Duong dan video dau ra.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Duong dan model YOLOv8 (.pt).")
    parser.add_argument("--conf", type=float, default=0.4, help="Nguong confidence detector.")
    parser.add_argument("--line-ratio", type=float, default=0.5, help="Vi tri line dem theo chieu cao frame.")
    parser.add_argument("--max-runtime-sec", type=int, default=28800, help="Gioi han thoi gian chay (giay).")
    parser.add_argument("--sort-iou-thresh", type=float, default=0.3, help="Nguong IoU de ghep track.")
    parser.add_argument("--sort-max-age", type=int, default=30, help="So frame track duoc giu khi mat detect.")
    parser.add_argument("--sort-min-hits", type=int, default=3, help="So lan cap nhat toi thieu de xac nhan track.")
    parser.add_argument("--hide-window", action="store_true", help="Khong hien thi cua so preview.")
    return parser.parse_args()


def parse_source(source_value):
    if isinstance(source_value, str) and source_value.isdigit():
        return int(source_value)
    return source_value


def cleanup_track_state(track_state, active_ids, current_frame_idx, stale_after_frames=90):
    stale_ids = []
    for track_id, state in track_state.items():
        if track_id in active_ids:
            continue
        if current_frame_idx - state["last_seen_frame"] > stale_after_frames:
            stale_ids.append(track_id)

    for track_id in stale_ids:
        del track_state[track_id]


def ve_thong_tin(frame, line_y, width, height, total_down, total_up):
    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "Line Count",
        (10, max(25, line_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    info_status = [("Enter", total_down), ("Exit", total_up), ("Total", total_down + total_up)]
    for i, (label, value) in enumerate(info_status):
        cv2.putText(
            frame,
            f"{label}: {value}",
            (10, height - ((i * 25) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


def iou_xyxy(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


class SortTrack:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = np.asarray(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits


class SortTracker:
    """
    SORT phien ban gon:
    - Ghep detection va track bang IoU + Hungarian matching
    - Giu track toi da max_age frame khi mat detect
    - Chi xuat track da duoc xac nhan qua min_hits
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.next_track_id = 1
        self.tracks = []
        self.frame_idx = 0

    def _build_iou_cost(self, detections):
        # Cost = 1 - IoU (cang nho cang tot)
        cost = np.ones((len(self.tracks), len(detections)), dtype=np.float32)
        for r, track in enumerate(self.tracks):
            for c, det in enumerate(detections):
                cost[r, c] = 1.0 - iou_xyxy(track.bbox, det[:4])
        return cost

    def update(self, detections):
        self.frame_idx += 1

        # Tang tuoi tat ca track truoc khi ghep frame moi
        for track in self.tracks:
            track.mark_missed()

        detections = detections if detections is not None else []
        detections = [np.asarray(det, dtype=np.float32) for det in detections]

        matched_track_idx = set()
        matched_det_idx = set()

        if self.tracks and detections:
            cost = self._build_iou_cost(detections)
            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                iou_score = 1.0 - float(cost[r, c])
                if iou_score < self.iou_threshold:
                    continue
                self.tracks[r].update(detections[c][:4])
                matched_track_idx.add(r)
                matched_det_idx.add(c)

        # Tao track moi cho detection chua duoc ghep
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_idx:
                continue
            new_track = SortTrack(track_id=self.next_track_id, bbox=det[:4])
            self.next_track_id += 1
            self.tracks.append(new_track)

        # Xoa track het han
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        # Chi tra track da cap nhat o frame hien tai + da duoc xac nhan
        active_tracks = []
        for track in self.tracks:
            if track.time_since_update != 0:
                continue
            if not track.is_confirmed(self.min_hits):
                continue
            active_tracks.append((track.track_id, track.bbox.copy()))
        return active_tracks


def people_counter_sort(
    source,
    output_path,
    model_path,
    conf=0.4,
    line_ratio=0.5,
    max_runtime_sec=28800,
    sort_iou_thresh=0.3,
    sort_max_age=30,
    sort_min_hits=3,
    show_window=True,
):
    if not (0.0 <= line_ratio <= 1.0):
        raise ValueError("--line-ratio phai nam trong [0.0, 1.0]")

    source = parse_source(source)
    output_path = Path(output_path)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model: {model_path}")

    if isinstance(source, str):
        source_path = Path(source)
        if not (source.startswith(("rtsp://", "http://", "https://")) or source_path.exists()):
            raise FileNotFoundError(f"Khong tim thay nguon video: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    disable_ultralytics_integration_callbacks()

    logger.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    logger.info("Opening source: %s", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {source}")

    sort_tracker = SortTracker(
        iou_threshold=sort_iou_thresh,
        max_age=sort_max_age,
        min_hits=sort_min_hits,
    )

    writer = None
    fps_counter = FPS().start()
    start_time = time.time()
    frame_idx = 0
    latencies_ms = []

    track_state = {}
    total_up = 0
    total_down = 0

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        output_fps = source_fps if source_fps and source_fps > 1 else 30.0

        if width <= 0 or height <= 0:
            raise RuntimeError("Khong doc duoc kich thuoc frame tu nguon video.")

        line_y = int(height * line_ratio)
        line_y = max(0, min(height - 1, line_y))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError(f"Khong tao duoc file output: {output_path}")

        logger.info(
            "Frame size: %dx%d | Source FPS: %.2f | Output FPS: %.2f",
            width,
            height,
            source_fps,
            output_fps,
        )
        logger.info(
            "Counting line Y = %d (ratio=%.2f) | SORT(iou=%.2f, max_age=%d, min_hits=%d)",
            line_y,
            line_ratio,
            sort_iou_thresh,
            sort_max_age,
            sort_min_hits,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or failed to read frame.")
                break

            frame_idx += 1
            active_ids = set()

            # Detect nguoi bang YOLOv8
            t_infer = time.perf_counter()
            results = model.predict(
                source=frame,
                classes=[0],
                conf=conf,
                verbose=False,
            )

            detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None

                if confs is None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.tolist()
                        detections.append([x1, y1, x2, y2, 1.0])
                else:
                    for box, det_conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box.tolist()
                        detections.append([x1, y1, x2, y2, float(det_conf)])

            tracks = sort_tracker.update(detections)
            latencies_ms.append((time.perf_counter() - t_infer) * 1000.0)

            for track_id, bbox in tracks:
                x1, y1, x2, y2 = map(int, bbox.tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                active_ids.add(track_id)

                state = track_state.get(track_id)
                if state is None:
                    track_state[track_id] = {
                        "last_cy": cy,
                        "counted": False,
                        "last_seen_frame": frame_idx,
                    }
                else:
                    prev_cy = state["last_cy"]
                    if not state["counted"]:
                        if prev_cy < line_y <= cy:
                            total_down += 1
                            state["counted"] = True
                        elif prev_cy > line_y >= cy:
                            total_up += 1
                            state["counted"] = True

                    state["last_cy"] = cy
                    state["last_seen_frame"] = frame_idx

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            cleanup_track_state(track_state, active_ids, frame_idx, stale_after_frames=90)
            ve_thong_tin(frame, line_y, width, height, total_down, total_up)

            writer.write(frame)

            if show_window:
                cv2.imshow("People Count (SORT)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("Stopped by user (ESC).")
                    break

            fps_counter.update()

            if max_runtime_sec > 0 and (time.time() - start_time) > max_runtime_sec:
                logger.info("Reached max runtime: %s seconds", max_runtime_sec)
                break

        fps_counter.stop()
        return {
            "elapsed_sec": fps_counter.elapsed(),
            "approx_fps": fps_counter.fps(),
            "latency_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
            "enter": total_down,
            "exit": total_up,
            "total": total_down + total_up,
            "frames": frame_idx,
            "output": str(output_path),
        }

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    summary = people_counter_sort(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        line_ratio=args.line_ratio,
        max_runtime_sec=args.max_runtime_sec,
        sort_iou_thresh=args.sort_iou_thresh,
        sort_max_age=args.sort_max_age,
        sort_min_hits=args.sort_min_hits,
        show_window=not args.hide_window,
    )

    logger.info("Elapsed time: %.2f sec", summary["elapsed_sec"])
    logger.info("Approx. FPS: %.2f", summary["approx_fps"])
    logger.info("Latency (ms/frame): %.2f", summary["latency_ms"])
    logger.info("Total Enter: %d", summary["enter"])
    logger.info("Total Exit: %d", summary["exit"])
    logger.info("Total Counted: %d", summary["total"])
    logger.info("Frames Processed: %d", summary["frames"])
    logger.info("Saved output: %s", summary["output"])


if __name__ == "__main__":
    main()
