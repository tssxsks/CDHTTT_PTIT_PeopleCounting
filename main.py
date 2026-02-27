import argparse
import logging
import time
from pathlib import Path

import cv2
from imutils.video import FPS
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
DEFAULT_INPUT_PATH = PROJECT_DIR / "Input" / "Test.mp4"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "Final_output.mp4"


def disable_ultralytics_integration_callbacks():
    """
    Ultralytics 8.0.110 tu dong import cac callback integration (TensorBoard, MLflow, ...)
    ngay ca khi chi predict/track. Neu moi truong co xung dot TensorFlow/TensorBoard/Numpy
    thi se vo ngay luc model.track().
    """
    try:
        from ultralytics.yolo.utils import callbacks as callbacks_pkg
        from ultralytics.yolo.utils.callbacks import base as callbacks_base

        def _noop_add_integration_callbacks(_instance):
            return None

        callbacks_base.add_integration_callbacks = _noop_add_integration_callbacks
        callbacks_pkg.add_integration_callbacks = _noop_add_integration_callbacks
        logger.info("Disabled Ultralytics integration callbacks for inference/tracking.")
    except Exception as exc:
        logger.warning("Could not disable Ultralytics integration callbacks: %s", exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="He thong dem so nguoi ra vao tu video (YOLOv8 + ByteTrack)"
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_INPUT_PATH),
        help="Duong dan video dau vao, camera index (0), hoac URL stream.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Duong dan video dau ra.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Duong dan model YOLOv8 (.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Nguong tin cay cho detector (mac dinh: 0.4).",
    )
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.5,
        help="Vi tri line dem theo chieu cao frame (0.0-1.0, mac dinh: 0.5).",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=int,
        default=28800,
        help="Gioi han thoi gian chay tinh bang giay (mac dinh: 8 gio).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="File cau hinh tracker cua Ultralytics (mac dinh: bytetrack.yaml).",
    )
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Khong hien thi cua so preview (phu hop server/headless).",
    )
    return parser.parse_args()


def parse_source(source_value):
    # Cho phep truyen camera index nhu "0", "1"
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


def people_counter(
    source,
    output_path,
    model_path,
    conf=0.4,
    line_ratio=0.5,
    max_runtime_sec=28800,
    tracker_cfg="bytetrack.yaml",
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

    # Tranh loi moi truong TensorBoard/TensorFlow khi Ultralytics tu import callback integrations
    disable_ultralytics_integration_callbacks()

    logger.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    logger.info("Opening source: %s", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {source}")

    writer = None
    fps_counter = FPS().start()
    start_time = time.time()
    frame_idx = 0
    latencies_ms = []

    # Luu trang thai theo tung track ID:
    # {
    #   track_id: {"last_cy": int, "counted": bool, "last_seen_frame": int}
    # }
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

        logger.info("Frame size: %dx%d | Source FPS: %.2f | Output FPS: %.2f", width, height, source_fps, output_fps)
        logger.info("Counting line Y = %d (ratio=%.2f)", line_y, line_ratio)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or failed to read frame.")
                break

            frame_idx += 1
            active_ids = set()

            # Tracking person (COCO class 0)
            t_infer = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=conf,
                tracker=tracker_cfg,
                verbose=False,
            )
            latencies_ms.append((time.perf_counter() - t_infer) * 1000.0)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_tensor = results[0].boxes.xyxy
                ids_tensor = results[0].boxes.id

                boxes = boxes_tensor.int().cpu().numpy()
                ids = ids_tensor.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
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
                            # Di xuong -> ENTER
                            if prev_cy < line_y <= cy:
                                total_down += 1
                                state["counted"] = True
                            # Di len -> EXIT
                            elif prev_cy > line_y >= cy:
                                total_up += 1
                                state["counted"] = True

                        state["last_cy"] = cy
                        state["last_seen_frame"] = frame_idx

                    # Ve bbox + ID + centroid
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

            # Ve line dem
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
                text = f"{label}: {value}"
                cv2.putText(
                    frame,
                    text,
                    (10, height - ((i * 25) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            writer.write(frame)

            if show_window:
                cv2.imshow("People Count", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("Stopped by user (ESC).")
                    break

            fps_counter.update()

            if max_runtime_sec > 0 and (time.time() - start_time) > max_runtime_sec:
                logger.info("Reached max runtime: %s seconds", max_runtime_sec)
                break

        fps_counter.stop()
        summary = {
            "elapsed_sec": fps_counter.elapsed(),
            "approx_fps": fps_counter.fps(),
            "latency_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
            "enter": total_down,
            "exit": total_up,
            "total": total_down + total_up,
            "frames": frame_idx,
            "output": str(output_path),
        }
        return summary

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    summary = people_counter(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        line_ratio=args.line_ratio,
        max_runtime_sec=args.max_runtime_sec,
        tracker_cfg=args.tracker,
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
