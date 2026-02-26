# He thong dem so nguoi ra vao tu video

Project nay su dung `Ultralytics YOLOv8` de phat hien nguoi (class `person`) va `ByteTrack` de gan ID theo tung frame, sau do dem so nguoi **di vao / di ra** khi cat qua mot duong dem trong video.

## Kien truc hien tai

- Detection: `YOLOv8` (`ultralytics`)
- Tracking: `ByteTrack` thong qua `model.track(..., tracker="bytetrack.yaml")`
- Counting: Kiem tra huong di chuyen cua tam bbox khi cat qua line dem
- Output: Hien thi preview + xuat video `mp4`

Luu y: Thu muc `tracker/` con chua `CentroidTracker` (ma cu/legacy), nhung `main.py` hien tai dang dung `ByteTrack` cua Ultralytics.

## Cai dat

```bash
pip install -r requirements.txt
```

Dat model vao thu muc project (mac dinh la `yolov8n.pt`).

## Chay nhanh

Mac dinh script doc video `Input/Test.mp4` va xuat `Final_output.mp4`:

```bash
python main.py
```

## Cach dung (tham so)

```bash
python main.py --source Input/Test.mp4 --output Final_output.mp4
```

Mot so tuy chon huu ich:

- `--source`: duong dan video, camera index (`0`), hoac URL stream (`rtsp://...`)
- `--output`: file video dau ra
- `--model`: model YOLO (`.pt`), mac dinh `yolov8n.pt`
- `--conf`: nguong confidence (mac dinh `0.4`)
- `--line-ratio`: vi tri line dem theo chieu cao frame (`0.5` = giua man hinh)
- `--max-runtime-sec`: gioi han thoi gian chay (mac dinh `28800` = 8 gio)
- `--hide-window`: tat preview (phu hop chay tren server/headless)

Vi du:

```bash
python main.py --source 0 --line-ratio 0.55
python main.py --source rtsp://user:pass@ip/stream --hide-window
```

## Dau ra

- Hien thi bbox + ID tung nguoi
- Ve duong dem ngang
- Hien thi so luong `Enter`, `Exit`, `Total`
- Luu video ket qua ra file `.mp4`

Nhan `Esc` de dung khi dang xem preview.

## Goi y cai thien tiep

- Them ROI (chi dem trong vung quan tam)
- Them nhieu line dem (ra/vao cho tung cua)
- Xuat log CSV theo thoi gian
- Dong goi thanh API hoac giao dien web
