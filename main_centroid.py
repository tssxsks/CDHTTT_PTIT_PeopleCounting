import argparse
import logging
import time
from pathlib import Path

import cv2
from imutils.video import FPS
from ultralytics import YOLO

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject


logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
DEFAULT_INPUT_PATH = PROJECT_DIR / "Input" / "Test.mp4"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "Final_output_centroid.mp4"


def disable_ultralytics_integration_callbacks():
    """
    Ultralytics 8.0.110 co the tu nap cac callback tich hop (TensorBoard, MLflow, ...)
    ngay ca khi chi du doan. Neu moi truong bi xung dot package thi se loi o model.predict().
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
        description="He thong dem so nguoi ra vao tu video (YOLOv8 + CentroidTracker)"
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
        help="Nguong tin cay detector (mac dinh: 0.4).",
    )
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.5,
        help="Vi tri line dem theo chieu cao frame (0.0-1.0).",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=int,
        default=28800,
        help="Gioi han thoi gian chay tinh bang giay (mac dinh: 8 gio).",
    )
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=50,
        help="So frame toi da cho phep mat doi tuong truoc khi xoa khoi tracker.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=50,
        help="Khoang cach toi da de ghep centroid giua hai frame.",
    )
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Khong hien thi cua so preview.",
    )
    return parser.parse_args()


def parse_source(source_value):
    # Cho phep nhap camera index duoi dang chuoi "0", "1", ...
    if isinstance(source_value, str) and source_value.isdigit():
        return int(source_value)
    return source_value


def ve_thong_tin(frame, line_y, width, height, total_down, total_up):
    # Ve line dem va cac chi so tong hop
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


def people_counter_centroid(
    source,
    output_path,
    model_path,
    conf=0.4,
    line_ratio=0.5,
    max_runtime_sec=28800,
    max_disappeared=50,
    max_distance=50,
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

    # Tranh loi moi truong khi Ultralytics tu nap callback tich hop
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

    centroid_tracker = CentroidTracker(
        maxDisappeared=max_disappeared,
        maxDistance=max_distance,
    )
    trackable_objects = {}
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
            "Counting line Y = %d (ratio=%.2f) | CentroidTracker(maxDisappeared=%d, maxDistance=%d)",
            line_y,
            line_ratio,
            max_disappeared,
            max_distance,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or failed to read frame.")
                break

            frame_idx += 1

            # Phat hien nguoi bang YOLOv8 (khong dung tracker tich hop)
            results = model.predict(
                source=frame,
                classes=[0],
                conf=conf,
                verbose=False,
            )

            rects = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.int().cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    rects.append((x1, y1, x2, y2))

            # Cap nhat tracker centroid tu cac bbox detect duoc
            objects = centroid_tracker.update(rects)

            # Tao bang tra bbox theo centroid de ve duoc box + ID
            rects_by_centroid = {}
            for (x1, y1, x2, y2) in rects:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                rects_by_centroid.setdefault((cx, cy), []).append((x1, y1, x2, y2))

            # Chi xu ly doi tuong dang xuat hien trong frame (disappeared == 0)
            for object_id, centroid in objects.items():
                if centroid_tracker.disappeared.get(object_id, 0) != 0:
                    continue

                centroid_tuple = (int(centroid[0]), int(centroid[1]))
                tracked_obj = trackable_objects.get(object_id)

                if tracked_obj is None:
                    tracked_obj = TrackableObject(object_id, centroid_tuple)
                else:
                    prev_centroid = tracked_obj.centroids[-1]

                    if not tracked_obj.counted:
                        # Di xuong -> ENTER
                        if prev_centroid[1] < line_y <= centroid_tuple[1]:
                            total_down += 1
                            tracked_obj.counted = True
                        # Di len -> EXIT
                        elif prev_centroid[1] > line_y >= centroid_tuple[1]:
                            total_up += 1
                            tracked_obj.counted = True

                    tracked_obj.centroids.append(centroid_tuple)
                    # Giu lich su centroid gon lai de tranh tang bo nho khi chay lau
                    if len(tracked_obj.centroids) > 32:
                        tracked_obj.centroids = tracked_obj.centroids[-32:]

                trackable_objects[object_id] = tracked_obj

                # Ve bbox neu tim duoc bbox co centroid trung khop
                rect_list = rects_by_centroid.get(centroid_tuple, [])
                if rect_list:
                    x1, y1, x2, y2 = rect_list.pop(0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_pos = (x1, max(20, y1 - 10))
                else:
                    label_pos = (centroid_tuple[0] - 10, max(20, centroid_tuple[1] - 10))

                # Ve tam va ID doi tuong
                cv2.circle(frame, centroid_tuple, 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID {object_id}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Don cac trackable object da bi xoa khoi CentroidTracker
            stale_trackable_ids = [
                object_id for object_id in trackable_objects.keys() if object_id not in centroid_tracker.objects
            ]
            for object_id in stale_trackable_ids:
                del trackable_objects[object_id]

            ve_thong_tin(frame, line_y, width, height, total_down, total_up)
            writer.write(frame)

            if show_window:
                cv2.imshow("People Count (CentroidTracker)", frame)
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
    summary = people_counter_centroid(
        source=args.source,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        line_ratio=args.line_ratio,
        max_runtime_sec=args.max_runtime_sec,
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
        show_window=not args.hide_window,
    )

    logger.info("Elapsed time: %.2f sec", summary["elapsed_sec"])
    logger.info("Approx. FPS: %.2f", summary["approx_fps"])
    logger.info("Total Enter: %d", summary["enter"])
    logger.info("Total Exit: %d", summary["exit"])
    logger.info("Total Counted: %d", summary["total"])
    logger.info("Frames Processed: %d", summary["frames"])
    logger.info("Saved output: %s", summary["output"])


if __name__ == "__main__":
    main()
