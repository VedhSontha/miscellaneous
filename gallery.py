import cv2
import torch
import numpy as np
import os
import time
from numpy.linalg import norm
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor

# --------------------------- parameters ---------------------------
SAVE_DIR = "target_person"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_IMAGES      = 100   # how many cropped images to save
FRAME_SKIP      = 1     # 1 = every frame, 2 = every 2nd frame, etc.
SIM_DIST_THRESH = 0.40  # smaller = stricter (distance after normalisation)

# ----------------------- model initialisation ---------------------
detector = YOLO("yolov5nu.pt")          # must be able to detect 'person'
extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="",
    device="cpu"
)

# -------------------------- helpers -------------------------------
def l2_normalise(v: np.ndarray) -> np.ndarray:
    """Return unit‑length version of vector v."""
    return v / (norm(v) + 1e-8)

def extract_person_embedding(image: np.ndarray, box) -> np.ndarray | None:
    """Crop ROI, resize, RGB‑convert, embed, L2‑normalise."""
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, (128, 256))
    roi = roi[:, :, ::-1]               # BGR ➜ RGB
    emb = extractor(roi)[0]
    return l2_normalise(emb)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance in [0, 2] for L2‑normalised vectors.
    (0 = identical, 1 = orthogonal, 2 = opposite.)
    """
    return 1.0 - float(np.dot(a, b))

# ----------------------- interactive capture ----------------------
selected_embedding = None
current_detections = []
current_frame      = None
count              = 0                 # how many images saved so far

def click_event(event, x, y, flags, param):
    """Left‑click a person box to set them as the target."""
    global selected_embedding
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in current_detections:
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                selected_embedding = extract_person_embedding(current_frame, box)
                if selected_embedding is not None:
                    print("✅ Target person selected.")
                break

cap = cv2.VideoCapture(0)
cv2.namedWindow("Capture Person")
cv2.setMouseCallback("Capture Person", click_event)

frame_id = 0
while cap.isOpened() and count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    current_frame = frame.copy()

    # ---------------- person detection ----------------
    results = detector.predict(source=frame, conf=0.4, imgsz=416, verbose=False)
    boxes = [
        tuple(map(int, det[:4]))
        for det in results[0].boxes.data.cpu().numpy()
        if int(det[5]) == 0          # class 0 = person
    ]
    current_detections = boxes

    # -------------- save crops of the target -----------
    if selected_embedding is not None:
        for box in current_detections:
            emb = extract_person_embedding(frame, box)
            if emb is None:
                continue
            dist = cosine_distance(selected_embedding, emb)
            if dist > SIM_DIST_THRESH:            # bigger = not similar
                continue

            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            save_path = os.path.join(SAVE_DIR, f"{count:03d}.jpg")
            cv2.imwrite(save_path, crop)
            count += 1
            print(f"[{count}/{MAX_IMAGES}] Saved: {save_path}")
            if count >= MAX_IMAGES:
                break

    # -------------- visualisation ----------------------
    for box in current_detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Capture Person", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------ compute mean embedding ------------------------
embeddings = []
for fname in os.listdir(SAVE_DIR):
    img_path = os.path.join(SAVE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (128, 256))
    img = img[:, :, ::-1]             # BGR ➜ RGB
    emb = extractor(img)[0]
    embeddings.append(l2_normalise(emb))

if embeddings:
    embeddings = np.stack(embeddings)
    mean_embedding = l2_normalise(embeddings.mean(axis=0))
    np.save("mean_embedding.npy", mean_embedding)
    print("✅ Mean embedding saved to mean_embedding.npy")
else:
    print("⚠️  No valid embeddings collected.")