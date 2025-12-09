import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import time
import os
import matplotlib.pyplot as plt
from collections import deque

# two set of config for two videos
# suit 3e format

# # -------------------------
# # CONFIG 1  30s-7:20s (440s-30s=dur6:50s)
# # -------------------------
# VIDEO_PATH = "upview.avi"
# CROP_SIZE = 180
# FRAME_STEP = 5
# SKIP_START_SEC = 30
# IGNORE_FROM_SEC = 30
# IGNORE_TO_SEC = 35
# USE_CENTER_CROP = False
# CUSTOM_X = 730 #1280
# CUSTOM_Y = 180 #800

# OUTPUT_DIR = "vis_upview"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# # -------------------------


# -------------------------
# CONFIG2 5s-6:55s (421s-5s=dur6:56s)
# -------------------------
VIDEO_PATH = "sideview.avi"
CROP_SIZE = 200
FRAME_STEP = 5
SKIP_START_SEC = 5
IGNORE_FROM_SEC = 5
IGNORE_TO_SEC = 25
USE_CENTER_CROP = False
CUSTOM_X = 580
CUSTOM_Y = 400

OUTPUT_DIR = "vis_sideview"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Globe parameters
GLOBE_RADIUS_PIXELS = 195
EDGE_EXCLUSION_ZONE = 40

# ORB parameters
REDUCED_ORB_FEATURES = 300
RESIZE_FACTOR = 0.5
USE_FAST_MATCHER = True

# Visualization
SHOW_VISUALIZATION = False
VISUALIZATION_FRAME_STEP = 10  # Show visualization every N frames
OUTPUT_VIDEO = None  # Set to filename to save video, None to disable

# -------------------------


def get_patch(frame):
    """Extract ROI patch"""
    h, w = frame.shape[:2]
    if USE_CENTER_CROP:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = CUSTOM_X, CUSTOM_Y

    patch = frame[
        cy - CROP_SIZE: cy + CROP_SIZE,
        cx - CROP_SIZE: cx + CROP_SIZE
    ]
    
    if RESIZE_FACTOR != 1.0:
        new_size = (int(patch.shape[1] * RESIZE_FACTOR), 
                    int(patch.shape[0] * RESIZE_FACTOR))
        patch = cv2.resize(patch, new_size, interpolation=cv2.INTER_LINEAR)
    
    return patch


def compute_match_score_fast(des_ref, des, bf):
    """Compute feature match score"""
    if des is None or des_ref is None or len(des) < 2:
        return 0
    
    if USE_FAST_MATCHER:
        matches = bf.match(des_ref, des)
        good = [m for m in matches if m.distance < 50]
        return len(good)
    else:
        matches = bf.knnMatch(des_ref, des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return len(good)


def visualize_frame(patch, kp, current_time, fps, frame_count, current_score, cycle_info):
    """Draw visualization with feature points and tracking info"""
    # Scale patch back to original size for visualization
    if RESIZE_FACTOR != 1.0:
        vis_size = (CROP_SIZE * 2, CROP_SIZE * 2)
        vis_patch = cv2.resize(patch, vis_size, interpolation=cv2.INTER_LINEAR)
    else:
        vis_patch = patch.copy()
    
    h, w = vis_patch.shape[:2]
    center = (w // 2, h // 2)
    
    # Scale radius for visualization
    vis_radius = int(GLOBE_RADIUS_PIXELS / RESIZE_FACTOR)
    vis_inner = int(30 / RESIZE_FACTOR)
    vis_outer = int((GLOBE_RADIUS_PIXELS - EDGE_EXCLUSION_ZONE) / RESIZE_FACTOR)
    
    # Draw circles
    cv2.circle(vis_patch, center, vis_radius, (100, 100, 100), 2)
    cv2.circle(vis_patch, center, 3, (255, 255, 255), -1)
    cv2.circle(vis_patch, center, vis_inner, (50, 50, 50), 1)
    cv2.circle(vis_patch, center, vis_outer, (50, 50, 50), 1)
    
    # Draw feature points
    if kp is not None and len(kp) > 0:
        for keypoint in kp:
            x, y = keypoint.pt
            # Scale coordinates if needed
            if RESIZE_FACTOR != 1.0:
                x = int(x / RESIZE_FACTOR)
                y = int(y / RESIZE_FACTOR)
            else:
                x, y = int(x), int(y)
            
            cv2.circle(vis_patch, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(vis_patch, (x, y), 5, (0, 255, 0), 1)
    
    # Add text info
    cv2.putText(vis_patch, f"Time: {current_time:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_patch, f"Frame: {frame_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_patch, f"Features: {len(kp) if kp else 0}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_patch, f"Match Score: {current_score}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_patch, f"FPS: {fps:.1f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if cycle_info and cycle_info['count'] > 0:
        cv2.putText(vis_patch, f"Cycles: {cycle_info['count']}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis_patch


def main():
    start_time = time.time()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0 or np.isnan(fps):
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = []
    frame_times = []

    start_frame = int(SKIP_START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, ref_frame = cap.read()
    if not ret:
        print("Error reading first frame")
        return

    ref_patch = get_patch(ref_frame)

    # Setup ORB
    orb = cv2.ORB_create(
        nfeatures=REDUCED_ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        patchSize=31,
        fastThreshold=20
    )
    
    kp_ref, des_ref = orb.detectAndCompute(ref_patch, None)
    
    if USE_FAST_MATCHER:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Save reference
    ref_vis = cv2.drawKeypoints(
        ref_patch, kp_ref, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(f"{OUTPUT_DIR}/reference_keypoints.jpg", ref_vis)

    # Setup video writer (disabled)
    out = None

    frame_number = start_frame
    frame_count = 0
    vis_frame_count = 0
    process_start = time.time()
    fps_history = deque(maxlen=30)
    
    cycle_info = {'count': 0, 'period': 0}

    print("Processing frames with visualization...")
    print("Press 'q' to quit")
    
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()
            
            patch = get_patch(frame)
            kp, des = orb.detectAndCompute(patch, None)

            score = compute_match_score_fast(des_ref, des, bf)

            scores.append(score)
            current_time = frame_number / fps
            frame_times.append(current_time)

            # Calculate FPS
            frame_process_time = time.time() - frame_start
            fps_history.append(1.0 / frame_process_time if frame_process_time > 0 else 0.0)
            avg_fps = float(np.mean(list(fps_history)))

            # Visualize only every VISUALIZATION_FRAME_STEP frames
            if SHOW_VISUALIZATION and (frame_count % VISUALIZATION_FRAME_STEP == 0):
                vis_frame = visualize_frame(patch, kp, current_time, 
                                           avg_fps, frame_count, score, cycle_info)
                cv2.imshow('ORB Feature Matching', vis_frame)
                vis_frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_number += FRAME_STEP
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - process_start
                samples_per_sec = frame_count / elapsed
                print(f"Processed {frame_count} samples, {samples_per_sec:.1f} samples/sec")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    process_time = time.time() - process_start
    avg_rate = frame_count / process_time
    print(f"\nProcessing complete: {frame_count} samples in {process_time:.2f}s")
    print(f"Displayed {vis_frame_count} visualization frames")
    print(f"Average processing rate: {avg_rate:.1f} samples/sec")

    scores = np.array(scores)
    times = np.array(frame_times)

    # Apply smoothing to scores
    if len(scores) > 11:
        scores_smooth = savgol_filter(scores, 11, 3)
    else:
        scores_smooth = scores

    # Peak detection
    peaks, properties = find_peaks(
        scores_smooth,
        distance=(fps / FRAME_STEP) * 0.5,
        prominence=np.max(scores_smooth) * 0.2,
    )

    peak_times = times[peaks]
    peak_frames = (peak_times * fps).astype(int)

    # Remove unwanted peaks
    valid_mask = ~((peak_times >= IGNORE_FROM_SEC) & (peak_times <= IGNORE_TO_SEC))
    valid_peak_times = peak_times[valid_mask]
    valid_peak_frames = peak_frames[valid_mask]

    # Calculate cycle periods
    if len(valid_peak_times) > 1:
        cycle_periods = np.diff(valid_peak_times)
        mean_period = np.mean(cycle_periods)
        print(f"\nDetected {len(valid_peak_times)} valid peaks")
        print(f"Peak times: {valid_peak_times}")
        print(f"Cycle periods: {cycle_periods}")
        print(f"Mean cycle period: {mean_period:.2f}s")
        print(f"Angular velocity: {360.0 / mean_period:.2f} deg/s")
        cycle_info['count'] = len(valid_peak_times) - 1
        cycle_info['period'] = mean_period
    elif len(valid_peak_times) == 1:
        print(f"\nDetected 1 peak at time: {valid_peak_times[0]:.1f}s")
        rotation_time = valid_peak_times[0] - SKIP_START_SEC
        print(f"Rotation cycle: {rotation_time:.2f}s")
        print(f"Angular velocity: {360.0 / rotation_time:.2f} deg/s")

    # Save peak visualizations
    cap = cv2.VideoCapture(VIDEO_PATH)
    for pf, pt in zip(valid_peak_frames, valid_peak_times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
        ret, frame = cap.read()
        if not ret:
            continue

        patch = get_patch(frame)
        kp, des = orb.detectAndCompute(patch, None)

        # Scale for visualization
        if RESIZE_FACTOR != 1.0:
            vis_size = (CROP_SIZE * 2, CROP_SIZE * 2)
            patch_vis = cv2.resize(patch, vis_size, interpolation=cv2.INTER_LINEAR)
        else:
            patch_vis = patch

        kp_img = cv2.drawKeypoints(
            patch_vis, kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(f"{OUTPUT_DIR}/peak_{pt:.1f}_keypoints.jpg", kp_img)

        # Match visualization
        if des is not None and des_ref is not None:
            if USE_FAST_MATCHER:
                matches = bf.match(des_ref, des)
                good = [m for m in matches if m.distance < 50]
            else:
                matches = bf.knnMatch(des_ref, des, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            match_img = cv2.drawMatches(
                ref_patch, kp_ref,
                patch, kp,
                good[:20],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(f"{OUTPUT_DIR}/peak_{pt:.1f}_matches.jpg", match_img)

    cap.release()

    # Create detailed plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Match scores over time
    ax1.plot(times, scores, linewidth=0.5, alpha=0.3, label='Raw', color='lightblue')
    ax1.plot(times, scores_smooth, linewidth=2, label='Smoothed', color='blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Match Score')
    ax1.set_title('Feature Match Score Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for t in valid_peak_times:
        ax1.axvline(t, color='red', linestyle='--', alpha=0.7)
        ax1.text(t, ax1.get_ylim()[1] * 0.95, f'{t:.1f}s', 
                rotation=90, va='top', fontsize=8)

    # Plot 2: Angular velocity (if multiple peaks detected)
    if len(valid_peak_times) > 1:
        cycle_periods = np.diff(valid_peak_times)
        angular_velocities = 360.0 / cycle_periods
        mid_times = (valid_peak_times[:-1] + valid_peak_times[1:]) / 2
        
        ax2.plot(mid_times, angular_velocities, 'o-', linewidth=2, markersize=8)
        ax2.axhline(np.mean(angular_velocities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(angular_velocities):.2f} deg/s')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (deg/s)')
        ax2.set_title('Angular Velocity Between Cycles')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/analysis.png", dpi=150)
    plt.close()

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()