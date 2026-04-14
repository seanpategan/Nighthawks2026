import requests, time, sqlite3, json, os, cv2, numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from PIL import Image
from io import BytesIO
from collections import deque

model = YOLO("yolov8m.pt")

CAMERAS = {
    "inbound":  "https://webcams.nyctmc.org/api/cameras/32651453-d9de-4cde-af4b-be42b8de2775/image",
    "outbound": "https://webcams.nyctmc.org/api/cameras/74ca7d62-c8e5-4986-82fa-cd8d8db835b9/image",
}
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
history = {name: deque(maxlen=20) for name in CAMERAS}

conn = sqlite3.connect('traffic.db')
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("""CREATE TABLE IF NOT EXISTS counts (
    timestamp TEXT, camera TEXT,
    cars INT, trucks INT, buses INT, motorcycles INT, total INT, rank INT
)""")
conn.commit()

def annotate(img_array, results):
    overlay = img_array.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colors(int(box.cls[0]), True), -1)
    out = cv2.addWeighted(overlay, 0.3, img_array, 0.7, 0)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls, conf = int(box.cls[0]), float(box.conf[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), colors(cls, True), 2)
        cv2.putText(out, f"{results[0].names[cls]} {conf:.2f}",
                    (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors(cls, True), 1)
    return out

def process(img, frame_path):
    results = model(img)
    counts = {c: 0 for c in CAR_CLASSES}
    for box in results[0].boxes:
        cls = int(box.cls)
        if cls in CAR_CLASSES:
            counts[cls] += 1
    cv2.imwrite(frame_path, annotate(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), results))
    return counts

def rank(total, hist):
    hist.append(total)
    if len(hist) < 5:
        return min(10, max(1, round((total / 20) * 10) or 1))
    lo, hi = min(hist), max(hist)
    return 5 if lo == hi else round(min(10, max(1, 1 + (total - lo) / (hi - lo) * 9)))

while True:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    combined = {"timestamp": ts}

    for name, url in CAMERAS.items():
        try:
            img = Image.open(BytesIO(requests.get(url, timeout=10).content))
            counts = process(img, f"latest_frame_{name}.jpg")
            total = sum(counts.values())
            r = rank(total, history[name])
            conn.execute("INSERT INTO counts VALUES (datetime('now'),?,?,?,?,?,?,?)",
                         (name, counts[2], counts[7], counts[5], counts[3], total, r))
            conn.commit()
            combined[name] = {"cars": counts[2], "motorcycles": counts[3],
                               "buses": counts[5], "trucks": counts[7], "total": total, "rank": r}
            print(f"{ts} [{name}] cars={counts[2]} moto={counts[3]} bus={counts[5]} truck={counts[7]} total={total} rank={r}")
        except Exception as e:
            conn.rollback()
            print(f"{ts} [{name}] error: {e}")

    with open("latest_stats.json.tmp", "w") as f:
        f.write(json.dumps(combined))
    os.replace("latest_stats.json.tmp", "latest_stats.json")
    time.sleep(10)
