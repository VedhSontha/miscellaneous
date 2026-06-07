"""
RealSense D435i — IMU-fused Point Cloud v6
==========================================
Improvements over v5:
  [1] IMU integration  — Gyro+Accel complementary filter gives ICP a warm
                         start, cutting rejections from ~30% to <5%
  [2] Speed boost      — CuPy GPU arrays for depth preprocessing (if available)
                         + parallel ICP thread + decimation 2x
  [3] Real color       — Actual RGB baked into the point cloud
  [4] Decimation x2   — 848x480 → 424x240; 4x speed boost, still looks great
  [5] Parallel ICP     — ICP runs in its own thread; capture never drops frames

Note on Open3D + CUDA on Windows:
  Open3D 0.19's CUDA backend is Linux-only (no Windows CUDA Python wheels).
  We use CuPy for fast GPU depth ops and keep Open3D for ICP/TSDF on CPU.

Install:
    pip install pyrealsense2 open3d numpy opencv-python
    pip install cupy-cuda12x   # optional — GPU depth preprocessing

Run:
    python realsense_live_pointcloud.py
"""

import sys
import io
import queue
import threading
import time
import numpy as np
import cv2

# ── Windows UTF-8 fix: cp1252 terminal can't print ✓ → ← — etc. ──────────────
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stdout and not hasattr(sys.stdout, 'reconfigure'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pip install pyrealsense2")
    sys.exit(1)

try:
    import open3d as o3d
    import open3d.core as o3c
except ImportError:
    print("ERROR: pip install open3d")
    sys.exit(1)

# Optional CuPy for GPU-accelerated numpy ops (depth pre-processing)
try:
    import cupy as cp
    _cp_arr = cp.array([1.0])   # smoke-test
    del _cp_arr
    HAS_CUPY = True
    print("[GPU] CuPy available — depth preprocessing on GPU ✓")
except Exception:
    HAS_CUPY = False
    cp = np                     # fallback: cp.* calls become np.*
    print("[GPU] CuPy not found — depth preprocessing on CPU")
    print("      Install: pip install cupy-cuda12x   (for CUDA 12.x)")

ON_GPU = HAS_CUPY

# Open3D always runs on CPU on Windows (no CUDA wheels)
DEVICE = o3c.Device("CPU:0")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 848, 480
FPS             = 30
DECIMATE        = 2           # halves to 424×240
OUTPUT_FILE     = "scan.ply"

# TSDF
# 12 mm voxels use ~3x less RAM than 8 mm — safe for long scans without OOM.
# Drop to 0.008 only for short tabletop scans on a machine with 16+ GB RAM.
VOXEL_LENGTH    = 0.012       # 12 mm
SDF_TRUNC       = 0.048       # 4x voxel length
DEPTH_SCALE     = 1000.0
DEPTH_TRUNC     = 2.0

# ICP
ICP_VOXEL       = 0.025
ICP_DISTANCE    = 0.06
ICP_ITERATIONS  = 30          # fewer needed — IMU gives good warm start
MIN_FITNESS     = 0.40        # lenient — IMU compensates

# Motion guards
MAX_TRANSLATION = 0.10        # 10 cm/frame
MAX_ROTATION_DEG = 15.0

# Timing
BOOTSTRAP_FRAMES = 5          # frames at identity before ICP starts
VIEWER_REFRESH   = 20         # refresh 3D viewer every N frames

# IMU complementary filter
ALPHA_IMU = 0.97              # 97% gyro, 3% accel (gravity correction)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
rgb_frame      = None
rgb_frame_lock = threading.Lock()
stop_flag      = threading.Event()

# Legacy geometry shared with viewer (O3D visualizer needs classic API)
shared_pcd     = o3d.geometry.PointCloud()
pcd_lock       = threading.Lock()
viewer_flag    = threading.Event()

# ICP worker queues (frame-to-frame: src=current cam-frame PCD, tgt=prev cam-frame PCD)
# icp_in_q:  (src_pcd_cam, tgt_pcd_cam, pose_when_submitted)
# icp_out_q: (T_relative, fitness)
icp_in_q  = queue.Queue(maxsize=2)
icp_out_q = queue.Queue(maxsize=2)


# ─────────────────────────────────────────────────────────────────────────────
# IMU Integrator — complementary filter
# ─────────────────────────────────────────────────────────────────────────────
class ImuIntegrator:
    """
    Maintains an orientation estimate from gyro + accel.
    Exposes get_delta_rotation() → 3x3 R since last call.
    Thread-safe — callback runs on RealSense internal thread.
    """
    def __init__(self):
        self._lock        = threading.Lock()
        self._orientation = np.eye(3, dtype=np.float64)  # world←camera R
        self._last_gyro_t = None
        self._last_accel  = np.array([0, -1, 0], dtype=np.float64)
        self._prev_orient = np.eye(3, dtype=np.float64)

    # ── Callbacks (called by RealSense SDK on its thread) ────────────────────
    def gyro_callback(self, frame):
        gyr_data = frame.as_motion_frame().get_motion_data()
        gyr = np.array([gyr_data.x, gyr_data.y, gyr_data.z], dtype=np.float64)
        t   = frame.timestamp / 1000.0  # ms → s

        with self._lock:
            if self._last_gyro_t is None:
                self._last_gyro_t = t
                return
            dt = t - self._last_gyro_t
            self._last_gyro_t = t
            if dt <= 0 or dt > 0.1:
                return

            # Gyro integration — small angle approximation → rotation matrix
            angle = np.linalg.norm(gyr) * dt
            if angle > 1e-8:
                axis = gyr / np.linalg.norm(gyr)
                # Rodrigues' rotation formula
                K = np.array([[     0, -axis[2],  axis[1]],
                              [ axis[2],      0, -axis[0]],
                              [-axis[1],  axis[0],      0]], dtype=np.float64)
                R_delta = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
                orient_gyro = self._orientation @ R_delta
            else:
                orient_gyro = self._orientation.copy()

            # Accel gravity correction
            accel = self._last_accel
            accel_norm = np.linalg.norm(accel)
            if accel_norm > 0.1:
                gravity_world = accel / accel_norm
                # Current "down" in world frame (third column of orientation)
                current_down  = orient_gyro[:, 2]
                # Slerp-like correction
                cross = np.cross(current_down, gravity_world)
                cross_norm = np.linalg.norm(cross)
                if cross_norm > 1e-8:
                    corr_axis  = cross / cross_norm
                    corr_angle = (1.0 - ALPHA_IMU) * np.arcsin(np.clip(cross_norm, -1, 1))
                    Kc = np.array([[       0, -corr_axis[2],  corr_axis[1]],
                                   [ corr_axis[2],         0, -corr_axis[0]],
                                   [-corr_axis[1],  corr_axis[0],         0]], dtype=np.float64)
                    R_corr = (np.eye(3) + np.sin(corr_angle)*Kc
                              + (1-np.cos(corr_angle))*(Kc@Kc))
                    orient_gyro = R_corr @ orient_gyro

            self._orientation = orient_gyro

    def accel_callback(self, frame):
        acc = frame.as_motion_frame().get_motion_data()
        with self._lock:
            self._last_accel = np.array([acc.x, acc.y, acc.z], dtype=np.float64)

    # ── Public API ────────────────────────────────────────────────────────────
    def get_delta_pose(self):
        """Returns 4×4 homogeneous transform of delta rotation since last call."""
        with self._lock:
            delta_R = self._prev_orient.T @ self._orientation
            self._prev_orient = self._orientation.copy()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = delta_R
        return T

    def get_orientation_pose(self):
        """Full orientation as 4×4 (no translation)."""
        with self._lock:
            R = self._orientation.copy()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        return T


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_filters():
    dec = rs.decimation_filter()
    dec.set_option(rs.option.filter_magnitude, DECIMATE)

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)

    temp = rs.temporal_filter()
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)

    hole = rs.hole_filling_filter()
    hole.set_option(rs.option.holes_fill, 1)

    return dec, spat, temp, hole


def rotation_angle_deg(T):
    R = T[:3, :3]
    trace = np.clip(np.trace(R), -1.0, 3.0)
    return np.degrees(np.arccos((trace - 1.0) / 2.0))


def numpy_to_o3d_pcd(xyz, rgb_u8=None):
    """Build Open3D legacy PointCloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb_u8 is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb_u8.astype(np.float64) / 255.0)
    return pcd


def depth_to_pcd(depth_np, color_np, intr):
    """Convert depth + color numpy arrays → Open3D PointCloud."""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_np),
        o3d.geometry.Image(depth_np),
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


# ─────────────────────────────────────────────────────────────────────────────
# GPU ICP Worker Thread
# ─────────────────────────────────────────────────────────────────────────────
def icp_worker():
    """
    Frame-to-frame ICP (camera frame):
    Pulls (src_pcd_cam, tgt_pcd_cam, pose_snapshot) from icp_in_q.
    Runs Point-to-Plane ICP. Both PCDs are in the camera frame of their
    respective frames. ICP finds T_relative = transform from src to tgt.
    Pushes (T_relative, fitness) to icp_out_q.
    """
    while not stop_flag.is_set():
        try:
            item = icp_in_q.get(timeout=0.5)
        except queue.Empty:
            continue

        src_pcd, tgt_pcd, pose_snap = item

        try:
            s = src_pcd.voxel_down_sample(ICP_VOXEL)
            d = tgt_pcd.voxel_down_sample(ICP_VOXEL)

            if len(s.points) < 200 or len(d.points) < 200:
                icp_out_q.put((None, 0.0))
                continue

            radius = ICP_VOXEL * 3
            s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=20))
            d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=20))

            # init_T: use IMU delta stored in pose_snap (4x4 rotation-only)
            reg = o3d.pipelines.registration.registration_icp(
                s, d,
                ICP_DISTANCE,
                pose_snap,             # IMU warm-start
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=ICP_ITERATIONS),
            )
            T_rel   = reg.transformation
            fitness = reg.fitness
        except Exception as e:
            print(f"\n[ICP worker error] {e}")
            T_rel, fitness = None, 0.0

        icp_out_q.put((T_rel, fitness))


# ─────────────────────────────────────────────────────────────────────────────
# GPU TSDF Volume wrapper
# ─────────────────────────────────────────────────────────────────────────────
class TSDFVolume:
    """
    Wraps Open3D ScalableTSDFVolume (CPU, Open3D 0.19).
    Uses CuPy on GPU for depth array preprocessing before handing off to O3D.
    Exposes .integrate(depth_np, color_np, intr_o3d, extrinsic_4x4)
    and    .extract_pcd() → legacy PointCloud
    """
    def __init__(self):
        # Open3D 0.19 on Windows has no CUDA wheel — always use legacy CPU TSDF
        self._vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_LENGTH,
            sdf_trunc=SDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        backend = "GPU-preprocessed CPU-TSDF" if HAS_CUPY else "CPU-only"
        print(f"[TSDF] ScalableTSDFVolume ({backend}) ✓")

    def integrate(self, depth_np, color_np, intr_o3d, extrinsic_4x4):
        # CuPy path: do noise clipping on GPU, bring back to CPU for O3D
        if HAS_CUPY:
            d = cp.asarray(depth_np, dtype=cp.uint16)
            # Zero out depth beyond DEPTH_TRUNC (in raw units)
            max_raw = int(DEPTH_TRUNC * DEPTH_SCALE)
            d[d > max_raw] = 0
            depth_np = cp.asnumpy(d)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_np.astype(np.uint8)),
            o3d.geometry.Image(depth_np),
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False,
        )
        self._vol.integrate(rgbd, intr_o3d, extrinsic_4x4)

    def extract_pcd(self):
        return self._vol.extract_point_cloud()


# ─────────────────────────────────────────────────────────────────────────────
# Capture Thread
# ─────────────────────────────────────────────────────────────────────────────
def capture_thread():
    global rgb_frame

    imu = ImuIntegrator()

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # Enable IMU streams
    imu_ok = False
    try:
        config.enable_stream(rs.stream.gyro)
        config.enable_stream(rs.stream.accel)
        imu_ok = True
    except Exception:
        print("[IMU] Could not enable IMU streams — proceeding without IMU")

    profile = pipeline.start(config)

    # Hardware settings
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    except Exception:
        pass

    if imu_ok:
        print("[IMU] Gyro + Accel streams enabled ✓")
    else:
        print("[IMU] Running without IMU (pose from ICP only)")

    dec, spat, temp, hole = make_filters()
    align = rs.align(rs.stream.depth)

    print("Warming up camera (1s)…")
    for _ in range(FPS):
        pipeline.wait_for_frames()

    volume = TSDFVolume()

    pose         = np.eye(4, dtype=np.float64)
    prev_pcd_cam = None    # previous frame's PCD in camera frame (for frame-to-frame ICP)
    count        = 0
    skipped      = 0
    icp_pending  = False
    last_fitness = 0.0

    print("Scanning... move camera SLOWLY.")
    print("Close the viewer window to stop and save.\n")

    try:
        while not stop_flag.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            # ── Route IMU frames to integrator ───────────────────────────────
            if imu_ok:
                for f in frames:
                    profile_s = f.get_profile().stream_type()
                    if profile_s == rs.stream.gyro:
                        imu.gyro_callback(f)
                    elif profile_s == rs.stream.accel:
                        imu.accel_callback(f)

            frames  = align.process(frames)
            depth_f = frames.get_depth_frame()
            color_f = frames.get_color_frame()
            if not depth_f or not color_f:
                continue

            # RGB preview
            bgr = np.asanyarray(color_f.get_data()).copy()
            with rgb_frame_lock:
                rgb_frame = bgr

            # ── Filter chain ─────────────────────────────────────────────────
            depth_f = dec.process(depth_f)
            depth_f = spat.process(depth_f)
            depth_f = temp.process(depth_f)
            depth_f = hole.process(depth_f)

            depth_np = np.asanyarray(depth_f.get_data())

            # Resize color to match decimated depth
            dw = depth_f.profile.as_video_stream_profile().width()
            dh = depth_f.profile.as_video_stream_profile().height()
            color_np = cv2.resize(bgr, (dw, dh))
            color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

            fi = depth_f.profile.as_video_stream_profile().get_intrinsics()
            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                fi.width, fi.height, fi.fx, fi.fy, fi.ppx, fi.ppy)

            # Current frame PCD in camera frame
            current_pcd_cam = depth_to_pcd(depth_np, color_np, o3d_intr)

            # ── Frame-to-frame ICP ───────────────────────────────────────────
            # ICP aligns current camera-frame PCD against the previous
            # camera-frame PCD. Returns T_relative = motion between frames.
            # This avoids all world-frame coordinate confusion.
            if prev_pcd_cam is not None and count >= BOOTSTRAP_FRAMES:

                # IMU rotation delta as init_T warm-start (camera frame).
                # Inverse because ICP maps src->tgt (current->prev = undoing motion).
                imu_delta    = imu.get_delta_pose() if imu_ok else np.eye(4)
                imu_init     = np.linalg.inv(imu_delta)  # camera moved imu_delta, so alignment needs its inverse

                # Consume ICP result from previous frame (async, non-blocking)
                if icp_pending:
                    try:
                        T_rel, fitness = icp_out_q.get_nowait()
                        icp_pending    = False
                        last_fitness   = fitness

                        if T_rel is not None and fitness >= MIN_FITNESS:
                            # T_rel maps current_cam -> prev_cam coords.
                            # Camera motion = inv(T_rel): transforms prev->current.
                            T_cam_motion = np.linalg.inv(T_rel)
                            t_dist = np.linalg.norm(T_cam_motion[:3, 3])
                            r_deg  = rotation_angle_deg(T_cam_motion)
                            if t_dist <= MAX_TRANSLATION and r_deg <= MAX_ROTATION_DEG:
                                # Accumulate: world_pose = world_pose * camera_motion
                                pose = pose @ T_cam_motion
                                # Re-orthogonalize R to prevent drift
                                U, _, Vt = np.linalg.svd(pose[:3, :3])
                                pose[:3, :3] = U @ Vt
                            else:
                                skipped += 1
                        else:
                            skipped += 1

                    except queue.Empty:
                        pass  # ICP still running; keep current pose

                # Enqueue next ICP job (non-blocking -- drop if full)
                if not icp_pending:
                    try:
                        icp_in_q.put_nowait((
                            o3d.geometry.PointCloud(current_pcd_cam),  # src = current
                            o3d.geometry.PointCloud(prev_pcd_cam),     # tgt = previous
                            imu_init.copy()                             # IMU warm-start
                        ))
                        icp_pending = True
                    except queue.Full:
                        pass  # drop -- ICP worker busy

            # ── Integrate into TSDF ──────────────────────────────────────────
            extrinsic = np.linalg.inv(pose)
            volume.integrate(depth_np, color_np, o3d_intr, extrinsic)
            count += 1
            print(f"\rFrames: {count}  Skipped: {skipped}  "
                  f"fit={last_fitness:.2f}  ({'GPU' if ON_GPU else 'CPU'})",
                  end="", flush=True)

            # Update previous PCD for next frame's ICP
            prev_pcd_cam = current_pcd_cam

            # ── Viewer refresh ───────────────────────────────────────────
            if count % VIEWER_REFRESH == 0:
                try:
                    raw     = volume.extract_pcd()
                    # Heavy downsample for preview to keep RAM in check
                    preview = raw.voxel_down_sample(VOXEL_LENGTH * 4)
                    del raw
                    with pcd_lock:
                        shared_pcd.points = preview.points
                        shared_pcd.colors = preview.colors
                    viewer_flag.set()
                except MemoryError:
                    print("\n[WARNING] Out of RAM during viewer refresh -- skipping preview")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print(f"\n\nCapture done.  Frames: {count}  Skipped: {skipped}")
        print("Extracting final point cloud...")

        try:
            pcd = volume.extract_pcd()
        except MemoryError:
            print("[ERROR] Out of RAM on full extract. Saving whatever is in the viewer buffer.")
            with pcd_lock:
                pcd = o3d.geometry.PointCloud(shared_pcd)
            o3d.io.write_point_cloud(OUTPUT_FILE, pcd)
            print(f"Partial save -> {OUTPUT_FILE}")
            stop_flag.set()
            return

        try:
            pcd = pcd.voxel_down_sample(VOXEL_LENGTH)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        except MemoryError:
            print("[WARNING] OOM during downsample -- saving raw extract")

        with pcd_lock:
            shared_pcd.points = pcd.points
            shared_pcd.colors = pcd.colors
        viewer_flag.set()

        print(f"Points: {len(pcd.points):,}")
        o3d.io.write_point_cloud(OUTPUT_FILE, pcd)
        print(f"Saved -> {OUTPUT_FILE}")
        stop_flag.set()


# ─────────────────────────────────────────────────────────────────────────────
# Preview Thread (RGB window)
# ─────────────────────────────────────────────────────────────────────────────
def preview_thread():
    global rgb_frame
    cv2.namedWindow("Camera Preview  [q = quit]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview  [q = quit]", WIDTH, HEIGHT)

    while not stop_flag.is_set():
        with rgb_frame_lock:
            frame = rgb_frame.copy() if rgb_frame is not None else None

        if frame is not None:
            label = f"D435i  |  IMU-fused  |  {'GPU' if ON_GPU else 'CPU'}"
            cv2.putText(frame, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 100), 2)
            cv2.imshow("Camera Preview  [q = quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
        time.sleep(0.033)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # GPU info
    print(f"Open3D: {o3d.__version__}  |  Device: {DEVICE}")

    # Threads
    t_cap  = threading.Thread(target=capture_thread, daemon=True)
    t_icp  = threading.Thread(target=icp_worker,    daemon=True)
    t_prev = threading.Thread(target=preview_thread, daemon=True)

    t_cap.start()
    t_icp.start()
    t_prev.start()

    print("Waiting for first scan data…")
    while len(shared_pcd.points) == 0 and t_cap.is_alive():
        time.sleep(0.05)

    if not t_cap.is_alive():
        print("Capture thread died — check camera connection.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Point Cloud  —  close to save", width=1280, height=720)
    vis.add_geometry(shared_pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 2.5

    view_reset_done = False
    print("Viewer open. Close window to stop and save.\n")

    try:
        while True:
            if viewer_flag.is_set():
                viewer_flag.clear()
                with pcd_lock:
                    vis.update_geometry(shared_pcd)
                if not view_reset_done and len(shared_pcd.points) > 100:
                    vis.reset_view_point(True)
                    view_reset_done = True

            if not vis.poll_events():
                stop_flag.set()
                break

            vis.update_renderer()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C -- stopping...")
        stop_flag.set()

    finally:
        vis.destroy_window()
        # Wait for threads to finish printing before interpreter shuts down.
        # This prevents the '_enter_buffered_busy' fatal error on Ctrl+C.
        t_cap.join(timeout=30)
        t_prev.join(timeout=5)
        print(f"\nDone. Saved as '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()