import cv2
import numpy as np
from scipy.signal import savgol_filter
import time
import os
import matplotlib.pyplot as plt
from collections import deque

# upview failed

# # -------------------------
# # CONFIG 1  30s-7:20s (440s-30s=dur6:50s)
# # -------------------------
VIDEO_PATH = "upview.avi"
CROP_SIZE = 180
FRAME_STEP = 5
SKIP_START_SEC = 30
IGNORE_FROM_SEC = 30
IGNORE_TO_SEC = 35
USE_CENTER_CROP = False
CUSTOM_X = 730 #1280
CUSTOM_Y = 180 #800

OUTPUT_DIR = "vis_upview"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# # -------------------------


# Globe parameters
GLOBE_RADIUS_PIXELS = 155  # Use the outer exclusion circle as the tracking boundary
EDGE_EXCLUSION_ZONE = 0  # No edge exclusion needed now

# ORB parameters
ORB_FEATURES = 200
RESIZE_FACTOR = 1.0  # Keep full resolution for better tracking

# Tracking parameters
HISTORY_LENGTH = 10  # Compare with frames up to N frames in the past
MIN_MATCH_DISTANCE = 60  # Maximum Hamming distance for good matches
VELOCITY_SMOOTHING_WINDOW = 10  # Median filter window size
MIN_RADIUS = 20  # Minimum distance from center for valid points

# Visualization
SHOW_VISUALIZATION = True
VISUALIZATION_UPDATE_STEP = 10  # Update visualization every N frames
TERMINAL_UPDATE_STEP = 10  # Print to terminal every N frames

# -------------------------


class FeatureTracker:
    """Track features and compute angular velocity from multi-frame matching"""
    
    def __init__(self):
        # ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=ORB_FEATURES,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        
        # Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # History storage: (timestamp, keypoints, descriptors, frame_number)
        self.history = deque(maxlen=HISTORY_LENGTH)
        
        # Velocity tracking
        self.velocity_history = deque(maxlen=VELOCITY_SMOOTHING_WINDOW)
        self.angular_velocities = []
        self.timestamps = []
        
        # Stats
        self.frame_count = 0
        
    def get_patch(self, frame):
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
    
    def compute_angular_displacement(self, pt1, pt2, center):
        """Compute signed angular displacement from pt1 to pt2 around center
        Returns (angle, weight) where weight accounts for distance from rotation axis
        """
        r1 = np.array(pt1) - center
        r2 = np.array(pt2) - center
        
        # Check valid radius
        dist1 = np.linalg.norm(r1)
        dist2 = np.linalg.norm(r2)
        
        # Points should be within the globe radius and not too close to center
        if dist1 < MIN_RADIUS or dist2 < MIN_RADIUS:
            return None, 0.0
        if dist1 > GLOBE_RADIUS_PIXELS or dist2 > GLOBE_RADIUS_PIXELS:
            return None, 0.0
        
        # Compute angle
        dot = np.dot(r1, r2)
        det = r1[0] * r2[1] - r1[1] * r2[0]
        angle = np.arctan2(det, dot)
        
        # Weight by distance from center (proxy for distance from rotation axis)
        # Points further from center are more reliable for angular velocity measurement
        avg_dist = (dist1 + dist2) / 2.0
        weight = avg_dist / GLOBE_RADIUS_PIXELS  # Normalized weight [0, 1]
        
        return angle, weight
    
    def match_with_history(self, current_kp, current_des, current_time, patch_shape):
        """Match current features with historical frames and compute angular velocity"""
        if len(self.history) == 0:
            return 0.0
        
        center = np.array([patch_shape[1] / 2.0, patch_shape[0] / 2.0])
        
        all_angular_velocities = []
        
        # Match with multiple past frames (skip very recent ones for stability)
        for past_time, past_kp, past_des, past_frame_num in list(self.history)[:-2]:  # Skip last 2 frames
            if past_des is None or current_des is None:
                continue
                
            dt = current_time - past_time
            if dt < 0.1:  # Need at least 0.1 second gap
                continue
            
            # Match descriptors
            matches = self.bf.knnMatch(past_des, current_des, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance and m.distance < MIN_MATCH_DISTANCE:
                        good_matches.append(m)
            
            if len(good_matches) < 3:
                continue
            
            # Compute angular displacements for all good matches with weights
            angular_displacements = []
            weights = []
            for m in good_matches:
                past_pt = past_kp[m.queryIdx].pt
                curr_pt = current_kp[m.trainIdx].pt
                
                angle, weight = self.compute_angular_displacement(past_pt, curr_pt, center)
                if angle is not None and abs(angle) < np.pi:  # Sanity check
                    angular_displacements.append(angle)
                    weights.append(weight)
            
            if len(angular_displacements) < 3:
                continue
            
            angular_displacements = np.array(angular_displacements)
            weights = np.array(weights)
            
            # Remove statistical outliers using IQR
            q1, q3 = np.percentile(angular_displacements, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (angular_displacements >= lower) & (angular_displacements <= upper)
            else:
                mask = np.ones(len(angular_displacements), dtype=bool)
            
            if np.sum(mask) >= 3:
                filtered_angles = angular_displacements[mask]
                filtered_weights = weights[mask]
                
                # Use weighted median for more accurate measurement
                # Points further from center (higher weight) are more reliable
                sorted_indices = np.argsort(filtered_angles)
                sorted_angles = filtered_angles[sorted_indices]
                sorted_weights = filtered_weights[sorted_indices]
                
                cumulative_weights = np.cumsum(sorted_weights)
                total_weight = cumulative_weights[-1]
                median_idx = np.searchsorted(cumulative_weights, total_weight / 2.0)
                weighted_median_angle = sorted_angles[min(median_idx, len(sorted_angles) - 1)]
                
                omega = weighted_median_angle / dt  # rad/s
                all_angular_velocities.append(omega)
        
        if len(all_angular_velocities) == 0:
            return 0.0
        
        # Take median across all time differences
        omega_rad = np.median(all_angular_velocities)
        return omega_rad
    
    def process_frame(self, frame, timestamp):
        """Process a frame and return angular velocity"""
        patch = self.get_patch(frame)
        if patch.size == 0:
            return 0.0, None, None
        
        # Detect features
        kp, des = self.orb.detectAndCompute(patch, None)
        
        if kp is None or des is None or len(kp) == 0:
            return 0.0, patch, None
        
        # Compute angular velocity from history
        omega_rad = self.match_with_history(kp, des, timestamp, patch.shape)
        omega_deg = np.degrees(omega_rad)
        
        # Store in history
        self.history.append((timestamp, kp, des, self.frame_count))
        
        # Smooth velocity
        self.velocity_history.append(omega_deg)
        smoothed_omega = np.median(list(self.velocity_history))
        
        # Store for analysis
        self.angular_velocities.append(smoothed_omega)
        self.timestamps.append(timestamp)
        
        self.frame_count += 1
        
        return smoothed_omega, patch, kp
    
    def visualize_frame(self, patch, kp, omega_deg, timestamp, fps):
        """Draw visualization with feature points"""
        if RESIZE_FACTOR != 1.0:
            vis_size = (CROP_SIZE * 2, CROP_SIZE * 2)
            vis_patch = cv2.resize(patch, vis_size, interpolation=cv2.INTER_LINEAR)
        else:
            vis_patch = patch.copy()
        
        h, w = vis_patch.shape[:2]
        center = (w // 2, h // 2)
        
        # Scale radius for visualization
        vis_radius = int(GLOBE_RADIUS_PIXELS / RESIZE_FACTOR)
        vis_min_radius = int(MIN_RADIUS / RESIZE_FACTOR)
        
        # Draw circles - only outer tracking boundary
        cv2.circle(vis_patch, center, vis_radius, (100, 100, 100), 2)
        cv2.circle(vis_patch, center, 3, (255, 255, 255), -1)
        cv2.circle(vis_patch, center, vis_min_radius, (80, 80, 80), 1)  # Min radius indicator
        
        # Draw feature points
        if kp is not None and len(kp) > 0:
            for keypoint in kp:
                x, y = keypoint.pt
                if RESIZE_FACTOR != 1.0:
                    x = int(x / RESIZE_FACTOR)
                    y = int(y / RESIZE_FACTOR)
                else:
                    x, y = int(x), int(y)
                
                # Check if point is in valid region
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if vis_min_radius < dist < vis_radius:
                    color = (0, 255, 0)  # Green for valid points
                else:
                    color = (100, 100, 100)  # Gray for excluded points
                
                cv2.circle(vis_patch, (x, y), 3, color, -1)
                cv2.circle(vis_patch, (x, y), 5, color, 1)
        
        # Add text info
        cv2.putText(vis_patch, f"Time: {timestamp:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_patch, f"Frame: {self.frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_patch, f"Features: {len(kp) if kp else 0}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_patch, f"History: {len(self.history)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_patch, f"Omega: {omega_deg:.2f} deg/s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_patch, f"FPS: {fps:.1f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        period = 360.0 / abs(omega_deg) if abs(omega_deg) > 0.01 else float('inf')
        period_str = f"{period:.2f}s" if period < 1000 else "inf"
        cv2.putText(vis_patch, f"Period: {period_str}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_patch


def main():
    start_time = time.time()
    
    tracker = FeatureTracker()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0 or np.isnan(fps):
        fps = 30.0
    
    print(f"Video FPS: {fps:.2f}")
    print(f"History length: {HISTORY_LENGTH} frames")
    print(f"Velocity smoothing: {VELOCITY_SMOOTHING_WINDOW} frames")
    print("\nStarting real-time angular velocity tracking...")
    print("Press 'q' to quit\n")
    
    # Skip to start
    start_frame = int(SKIP_START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_number = start_frame
    process_times = deque(maxlen=30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            current_time = frame_number / fps
            
            # Process frame (always in real-time)
            omega_deg, patch, kp = tracker.process_frame(frame, current_time)
            
            frame_process_time = time.time() - frame_start
            process_times.append(frame_process_time)
            avg_fps = 1.0 / np.mean(list(process_times)) if len(process_times) > 0 else 0.0
            
            # Print to terminal every N frames
            if tracker.frame_count % TERMINAL_UPDATE_STEP == 0:
                period = 360.0 / abs(omega_deg) if abs(omega_deg) > 0.01 else float('inf')
                period_str = f"{period:.2f}s" if period < 1000 else "inf"
                print(f"Frame {tracker.frame_count:4d} | Time: {current_time:6.1f}s | "
                      f"Omega: {omega_deg:7.2f} deg/s | Period: {period_str:>8s} | "
                      f"Features: {len(kp) if kp is not None else 0:3d} | "
                      f"FPS: {avg_fps:.1f}")
            
            # Update visualization every N frames
            if SHOW_VISUALIZATION and (tracker.frame_count % VISUALIZATION_UPDATE_STEP == 0):
                if patch is not None:
                    vis_frame = tracker.visualize_frame(patch, kp, omega_deg, 
                                                       current_time, avg_fps)
                    cv2.imshow('Feature Tracking', vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_number += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Total frames: {tracker.frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {tracker.frame_count / total_time:.1f}")
    
    # Analysis
    velocities = np.array(tracker.angular_velocities)
    times = np.array(tracker.timestamps)
    
    # Remove initial transient (first few frames)
    if len(velocities) > 20:
        velocities = velocities[10:]
        times = times[10:]
    
    # Filter out near-zero velocities
    valid_mask = np.abs(velocities) > 0.1
    if np.sum(valid_mask) > 0:
        valid_velocities = velocities[valid_mask]
        mean_omega = np.mean(valid_velocities)
        std_omega = np.std(valid_velocities)
        mean_period = 360.0 / np.mean(np.abs(valid_velocities))
        
        print(f"\n{'='*70}")
        print(f"STATISTICS:")
        print(f"  Mean angular velocity: {mean_omega:.2f} ± {std_omega:.2f} deg/s")
        print(f"  Mean rotation period: {mean_period:.2f} s")
        print(f"  Velocity range: [{np.min(valid_velocities):.2f}, {np.max(valid_velocities):.2f}] deg/s")
        print(f"{'='*70}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Apply additional smoothing for plot
    if len(velocities) > 51:
        velocities_smooth = savgol_filter(velocities, 51, 3)
    else:
        velocities_smooth = velocities
    
    # Plot 1: Angular velocity over time
    ax1.plot(times, velocities, linewidth=0.5, alpha=0.3, label='Raw', color='lightblue')
    ax1.plot(times, velocities_smooth, linewidth=2, label='Smoothed', color='blue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angular Velocity (deg/s)', fontsize=12)
    ax1.set_title('Real-time Angular Velocity Tracking (Multi-frame Matching)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Period over time
    periods = np.array([360.0 / abs(v) if abs(v) > 0.01 else np.nan for v in velocities_smooth])
    ax2.plot(times, periods, linewidth=2, color='green')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Period (s/rotation)', fontsize=12)
    ax2.set_title('Rotation Period Over Time', fontsize=14)
    ax2.set_ylim([0, min(50, np.nanmax(periods) if not np.isnan(np.nanmax(periods)) else 50)])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/angular_velocity_analysis.png', dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR}/angular_velocity_analysis.png")
    
    plt.close()


if __name__ == "__main__":
    main()