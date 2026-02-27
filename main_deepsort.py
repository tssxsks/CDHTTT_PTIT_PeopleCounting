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
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "Final_output_deepsort.mp4"


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
        description="He thong dem so nguoi ra vao tu video (YOLOv8 + DeepSORT)"
    )
    parser.add_argument("--source", default=str(DEFAULT_INPUT_PATH), help="Video/camera/stream dau vao.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Duong dan video dau ra.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Duong dan model YOLOv8 (.pt).")
    parser.add_argument("--conf", type=float, default=0.4, help="Nguong confidence detector.")
    parser.add_argument("--line-ratio", type=float, default=0.5, help="Vi tri line dem theo chieu cao frame.")
    parser.add_argument("--max-runtime-sec", type=int, default=28800, help="Gioi han thoi gian chay (giay).")
    parser.add_argument("--max-age", type=int, default=30, help="So frame track duoc giu khi mat detect.")
    parser.add_argument("--n-init", type=int, default=3, help="So frame can de xac nhan track.")
    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=0.3,
        help="Nguong khoang cach cosine cho matching cua DeepSORT.",
    )
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


def create_deepsort_tracker(max_age, n_init, max_cosine_distance):
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as exc:
        raise ImportError(
            "Thieu thu vien deep-sort-realtime. Cai dat bang lenh: pip install deep-sort-realtime"
        ) from exc

    return DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
    )


def people_counter_deepsort(
    source,
    output_path,
    model_path,
    conf=0.4,
    line_ratio=0.5,
    max_runtime_sec=28800,
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,
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
    deepsort_tracker = create_deepsort_tracker(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
    )

    logger.info("Opening source: %s", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {source}")

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
            "Counting line Y = %d (ratio=%.2f) | DeepSORT(max_age=%d, n_init=%d, max_cosine_distance=%.2f)",
            line_y,
            line_ratio,
            max_age,
            n_init,
            max_cosine_distance,
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

            deep_sort_detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for box, det_conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box.tolist()
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    # Dinh dang detection cho deep-sort-realtime: ([left, top, width, height], confidence, class_name)
                    deep_sort_detections.append(([x1, y1, w, h], float(det_conf), "person"))

            tracks = deepsort_tracker.update_tracks(deep_sort_detections, frame=frame)
            latencies_ms.append((time.perf_counter() - t_infer) * 1000.0)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                if track.time_since_update > 0:
                    continue

                track_id = int(track.track_id)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
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
                cv2.imshow("People Count (DeepSORT)", frame)
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
    summary = people_counter_deepsort(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        line_ratio=args.line_ratio,
        max_runtime_sec=args.max_runtime_sec,
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
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
