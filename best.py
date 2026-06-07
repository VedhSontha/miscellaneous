import cv2
import os
import subprocess
import shutil
import open3d as o3d
import time

# === CONFIG ===
image_dir = "colmap_images"
output_dir = "colmap_output"
num_images = 30
capture_duration_sec = 10  # Increased to reduce blur
interval = capture_duration_sec / num_images
colmap_bin = r"C:\Users\vedhr\CODES\COLMAP\bin\colmap.exe"

# === CLEANUP ===
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(image_dir)
os.makedirs(output_dir)

# === WEBCAM CAPTURE WITH TRIGGER ===
print(f"[INFO] Webcam live. Press 's' to start auto-capture of {num_images} images in {capture_duration_sec} seconds.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam could not be opened.")
    exit()

count = 0
capture_started = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed.")
        break

    height, width, _ = frame.shape
    cv2.putText(frame, f"Resolution: {width}x{height}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if not capture_started:
        cv2.putText(frame, "Press 's' to start auto-capture", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frame, f"Capturing: {count}/{num_images}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if not capture_started and key == ord('s'):
        print("[INFO] Starting auto-capture...")
        capture_started = True
        start_time = time.time()

    if capture_started:
        elapsed = time.time() - start_time
        if count < num_images and elapsed >= count * interval:
            filename = os.path.join(image_dir, f"img_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Captured: {filename}")
            count += 1
        elif count >= num_images:
            print("[✅] Done capturing.")
            break

    if key == ord('q'):
        print("[INFO] Quit by user.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# === VERIFY IMAGE COUNT ===
actual_images = len(os.listdir(image_dir))
if actual_images < 10:
    print(f"[❌] Only {actual_images} images captured. Need at least 10.")
    exit()

# === COLMAP PATHS ===
database_path = os.path.join(output_dir, "database.db")
sparse_path = os.path.join(output_dir, "sparse")
dense_path = os.path.join(output_dir, "dense")
fused_ply = os.path.join(dense_path, "fused.ply")

# === COLMAP PIPELINE ===
def run_colmap():
    print("[INFO] Running feature extraction...")
    subprocess.run([
        colmap_bin, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1"
    ], check=True)

    print("[INFO] Matching features...")
    subprocess.run([
        colmap_bin, "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "1"
    ], check=True)

    print("[INFO] Building sparse map...")
    os.makedirs(sparse_path, exist_ok=True)
    subprocess.run([
        colmap_bin, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_path
    ], check=True)

    print("[INFO] Undistorting images...")
    os.makedirs(dense_path, exist_ok=True)
    subprocess.run([
        colmap_bin, "image_undistorter",
        "--image_path", image_dir,
        "--input_path", os.path.join(sparse_path, "0"),
        "--output_path", dense_path,
        "--output_type", "COLMAP"
    ], check=True)

    print("[INFO] Running dense stereo...")
    subprocess.run([
        colmap_bin, "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ], check=True)

    print("[INFO] Fusing depth maps...")
    subprocess.run([
        colmap_bin, "stereo_fusion",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply
    ], check=True)

try:
    run_colmap()
    print(f"[✅] Point cloud saved to: {fused_ply}")
except subprocess.CalledProcessError as e:
    print(f"[❌] COLMAP failed: {e}")
    exit()

# === VISUALIZE RESULT ===
if os.path.exists(fused_ply):
    print("[INFO] Visualizing fused point cloud...")
    pcd = o3d.io.read_point_cloud(fused_ply)
    o3d.visualization.draw_geometries([pcd])
else:
    print("[❌] Fused point cloud not found.")
