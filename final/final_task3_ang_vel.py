import cv2
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# sideview version
# current vel is lower than expected
# points' speed are not consistent-- not same radius at everywhere
# 4b surface linear velocity???
# wrong key points in the square

# -------------------------
# CONFIGURATION
# -------------------------
VIDEO_PATH = "sideview.avi"  # Set to 0 for webcam
CROP_SIZE = 240
USE_CENTER_CROP = False
CUSTOM_X = 580
CUSTOM_Y = 400

# Globe parameters - FIXED, not updated
GLOBE_RADIUS_PIXELS = 195

# Tracking parameters
MAX_FEATURES = 150
OPTICAL_FLOW_WIN_SIZE = 15
OPTICAL_FLOW_MAX_LEVEL = 2
MIN_TRACKING_POINTS = 50
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE = 10

# Point refresh strategy
REFRESH_EVERY_N_FRAMES = 300
POINT_AGE_THRESHOLD = 900
EDGE_EXCLUSION_ZONE = 40

# Cycle detection parameters
CYCLE_MATCH_THRESHOLD = 0.5
MIN_ROTATION_FOR_CYCLE = 300

# Angular velocity smoothing - INCREASED for smoother curves
VELOCITY_HISTORY_SIZE = 30  # Increased from 15
ANGLE_OUTLIER_THRESHOLD = 0.3  # Reject angles larger than this (radians)

# Display settings
SHOW_VISUALIZATION = True
OUTPUT_VIDEO = "output_tracked.avi"
SAVE_PLOT = True

# -------------------------


def ensure_points_array(pts):
    """Return an Nx2 array for points. pts can be None, (N,1,2), (N,2), or list."""
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)
    elif pts.ndim == 2 and pts.shape[1] == 2:
        pass
    elif pts.ndim == 1 and pts.size == 2:
        pts = pts.reshape(1, 2)
    else:
        pts = pts.reshape(-1, 2)
    return pts.astype(np.float32)


class AngularVelocityTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_points = np.zeros((0, 2), dtype=np.float32)
        self.point_ages = []

        # Enhanced smoothing
        self.velocity_history = deque(maxlen=VELOCITY_HISTORY_SIZE)
        self.fps_history = deque(maxlen=30)
        
        # Store raw velocities for better smoothing
        self.raw_velocities = []
        self.raw_timestamps = []

        # Reference for cycle detection
        self.reference_gray = None
        self.reference_descriptors = None
        self.cumulative_rotation = 0.0
        self.cycles_completed = 0
        self.last_cycle_time = 0.0
        self.cycle_times = []

        # Stats
        self.frame_times = []
        self.angular_velocities = []
        self.rotation_history = []
        self.frame_count = 0
        self.frames_since_refresh = 0

        # Feature & lk params
        self.feature_params = dict(
            maxCorners=MAX_FEATURES,
            qualityLevel=FEATURE_QUALITY,
            minDistance=FEATURE_MIN_DISTANCE,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(OPTICAL_FLOW_WIN_SIZE, OPTICAL_FLOW_WIN_SIZE),
            maxLevel=OPTICAL_FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # ORB for cycle detection
        self.orb = cv2.ORB_create(nfeatures=400)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_patch(self, frame):
        """Extract ROI patch and offset"""
        h, w = frame.shape[:2]
        if USE_CENTER_CROP:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = CUSTOM_X, CUSTOM_Y

        x1 = int(max(0, cx - CROP_SIZE))
        x2 = int(min(w, cx + CROP_SIZE))
        y1 = int(max(0, cy - CROP_SIZE))
        y2 = int(min(h, cy + CROP_SIZE))

        patch = frame[y1:y2, x1:x2]
        return patch, (x1, y1)

    def detect_features_excluding_edges(self, gray):
        """Detect features with exclusion zones"""
        h, w = gray.shape
        if w <= 2 * EDGE_EXCLUSION_ZONE or h <= 2 * EDGE_EXCLUSION_ZONE:
            mask = None
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[EDGE_EXCLUSION_ZONE:h - EDGE_EXCLUSION_ZONE,
                 EDGE_EXCLUSION_ZONE:w - EDGE_EXCLUSION_ZONE] = 255
            cv2.circle(mask, (w // 2, h // 2), 30, 0, -1)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        pts = ensure_points_array(pts)
        return pts

    def add_new_points(self, gray, existing_points, existing_ages):
        """Add new feature points while avoiding duplicates"""
        new_pts = self.detect_features_excluding_edges(gray)
        if new_pts.shape[0] == 0:
            return existing_points, existing_ages

        if existing_points.shape[0] == 0:
            ages = [0] * new_pts.shape[0]
            limited = new_pts[:MAX_FEATURES]
            return limited, ages[:limited.shape[0]]

        filtered_new = []
        for p in new_pts:
            dists = np.linalg.norm(existing_points - p, axis=1)
            if np.all(dists >= FEATURE_MIN_DISTANCE * 2):
                filtered_new.append(p)
        
        if len(filtered_new) == 0:
            return existing_points, existing_ages

        filtered_new = np.array(filtered_new, dtype=np.float32).reshape(-1, 2)
        combined = np.vstack([existing_points, filtered_new])
        combined_ages = existing_ages + [0] * filtered_new.shape[0]

        if combined.shape[0] > MAX_FEATURES:
            combined = combined[:MAX_FEATURES]
            combined_ages = combined_ages[:MAX_FEATURES]

        return combined, combined_ages

    @staticmethod
    def compute_tangential_displacement(p0, p1, center):
        """Compute signed angle (radians) the point moved around center"""
        p0 = np.asarray(p0).reshape(2,)
        p1 = np.asarray(p1).reshape(2,)
        center = np.asarray(center).reshape(2,)

        r0 = p0 - center
        r1 = p1 - center

        if np.linalg.norm(r0) < 1e-6 or np.linalg.norm(r1) < 1e-6:
            return 0.0

        dot = float(np.dot(r0, r1))
        det = float(r0[0] * r1[1] - r0[1] * r1[0])
        angle = np.arctan2(det, dot)
        return angle

    def compute_angular_velocity(self, prev_pts, curr_pts, dt, patch_shape):
        """Compute angular velocity with improved outlier rejection"""
        if prev_pts.shape[0] < 5 or dt <= 0:  # Require more points for stability
            return 0.0, 0.0, float('inf')

        center = np.array([patch_shape[1] / 2.0, patch_shape[0] / 2.0], dtype=np.float32)
        angular_displacements = []
        weights = []

        for p0, p1 in zip(prev_pts, curr_pts):
            dist_from_center = np.linalg.norm(p0 - center)
            
            # Only consider points in the optimal radius range
            if 60 < dist_from_center < GLOBE_RADIUS_PIXELS - 20:
                angle = self.compute_tangential_displacement(p0, p1, center)
                
                # Reject obvious outliers
                if abs(angle) < ANGLE_OUTLIER_THRESHOLD:
                    angular_displacements.append(angle)
                    # Weight by distance from center (points further out are more reliable)
                    weight = dist_from_center / GLOBE_RADIUS_PIXELS
                    weights.append(weight)

        if len(angular_displacements) < 5:
            return 0.0, 0.0, float('inf')

        # Use weighted median for robustness
        angular_displacements = np.array(angular_displacements)
        weights = np.array(weights)
        
        # Remove statistical outliers using IQR method
        q1, q3 = np.percentile(angular_displacements, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (angular_displacements >= lower_bound) & (angular_displacements <= upper_bound)
        
        if np.sum(mask) < 3:
            median_angle = np.median(angular_displacements)
        else:
            filtered_angles = angular_displacements[mask]
            median_angle = np.median(filtered_angles)

        omega_rad_per_sec = median_angle / dt
        omega_deg_per_sec = np.degrees(omega_rad_per_sec)

        if abs(omega_rad_per_sec) > 1e-6:
            period_sec = (2 * np.pi) / abs(omega_rad_per_sec)
        else:
            period_sec = float('inf')

        return omega_rad_per_sec, omega_deg_per_sec, period_sec

    def set_reference_frame(self, gray, timestamp):
        """Set reference frame for cycle detection"""
        self.reference_gray = gray.copy()
        kp_ref, des_ref = self.orb.detectAndCompute(self.reference_gray, None)
        self.reference_descriptors = des_ref
        self.last_cycle_time = timestamp
        self.cumulative_rotation = 0.0

    def check_cycle_completion(self, current_gray, current_time):
        """Check if a full rotation cycle is complete"""
        if self.reference_descriptors is None or self.reference_descriptors.size == 0:
            return False
        if self.cumulative_rotation < MIN_ROTATION_FOR_CYCLE:
            return False

        kp_curr, des_curr = self.orb.detectAndCompute(current_gray, None)
        if des_curr is None or des_curr.size == 0:
            return False

        matches = self.bf.match(self.reference_descriptors, des_curr)
        if len(self.reference_descriptors) == 0:
            return False

        match_ratio = len(matches) / float(len(self.reference_descriptors))
        if match_ratio >= CYCLE_MATCH_THRESHOLD:
            self.cycles_completed += 1
            cycle_duration = current_time - self.last_cycle_time
            self.cycle_times.append((current_time, cycle_duration))
            self.last_cycle_time = current_time
            self.cumulative_rotation = 0.0
            print(f"\n*** CYCLE COMPLETED at t={current_time:.2f}s: match_ratio={match_ratio:.2f}, duration={cycle_duration:.2f}s ***")
            return True
        return False

    def visualize_tracking(self, frame, prev_pts, curr_pts, ages, omega_deg, period, cycle_detected):
        """Draw tracking visualization"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Draw globe circle (FIXED RADIUS)
        cv2.circle(frame, center, GLOBE_RADIUS_PIXELS, (100, 100, 100), 1)
        cv2.circle(frame, center, 3, (255, 255, 255), -1)

        # Exclusion zone
        cv2.rectangle(frame, (EDGE_EXCLUSION_ZONE, EDGE_EXCLUSION_ZONE),
                      (w - EDGE_EXCLUSION_ZONE, h - EDGE_EXCLUSION_ZONE),
                      (50, 50, 50), 1)

        N = prev_pts.shape[0]
        for i in range(N):
            p0 = prev_pts[i]
            p1 = curr_pts[i]
            age = ages[i] if i < len(ages) else 0

            p0_int = (int(round(p0[0])), int(round(p0[1])))
            p1_int = (int(round(p1[0])), int(round(p1[1])))

            age_ratio = min(age / float(POINT_AGE_THRESHOLD), 1.0)
            color = (0, int(round(255 * (1 - age_ratio))), int(round(255 * age_ratio)))

            cv2.line(frame, p0_int, p1_int, (0, 255, 0), 1)
            cv2.circle(frame, p1_int, 3, color, -1)

        period_str = f"{period:.2f}" if period < 1e6 else "inf"
        cv2.putText(frame, f"Omega: {omega_deg:.2f} deg/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Period: {period_str} s/rot", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Points: {N}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Rotation: {self.cumulative_rotation:.1f} deg", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Cycles: {self.cycles_completed}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if cycle_detected:
            cv2.putText(frame, "*** CYCLE COMPLETE ***", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

        return frame

    def process_frame(self, frame, timestamp):
        """Process a single frame"""
        patch, offset = self.get_patch(frame)
        if patch.size == 0:
            return 0.0, float('inf'), frame, False

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_points = self.detect_features_excluding_edges(gray)
            self.point_ages = [0] * self.prev_points.shape[0]
            self.prev_timestamp = timestamp
            self.set_reference_frame(gray, timestamp)
            vis = self.visualize_tracking(patch.copy(),
                                          self.prev_points, self.prev_points,
                                          self.point_ages, 0.0, float('inf'), False)
            return 0.0, float('inf'), vis, False

        cycle_detected = False
        smoothed_omega = 0.0
        smoothed_period = float('inf')

        if self.prev_points.shape[0] > 0:
            p0_for_lk = self.prev_points.reshape(-1, 1, 2)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0_for_lk, None, **self.lk_params
            )

            status = None if status is None else status.reshape(-1)
            next_pts = ensure_points_array(next_pts)

            if status is None or next_pts.shape[0] == 0:
                good_prev = np.zeros((0, 2), dtype=np.float32)
                good_curr = np.zeros((0, 2), dtype=np.float32)
                good_ages = []
            else:
                mask = (status == 1)
                good_prev = self.prev_points[mask]
                good_curr = next_pts[mask]
                good_ages = [age for age, m in zip(self.point_ages, mask) if m]

                valid_indices = []
                h_patch, w_patch = gray.shape
                for i, pt in enumerate(good_curr):
                    x, y = pt
                    in_valid_region = (x >= EDGE_EXCLUSION_ZONE and x <= (w_patch - EDGE_EXCLUSION_ZONE) and
                                       y >= EDGE_EXCLUSION_ZONE and y <= (h_patch - EDGE_EXCLUSION_ZONE))
                    not_too_old = (good_ages[i] < POINT_AGE_THRESHOLD)
                    if in_valid_region and not_too_old:
                        valid_indices.append(i)

                if len(valid_indices) > 0:
                    good_prev = good_prev[valid_indices]
                    good_curr = good_curr[valid_indices]
                    good_ages = [good_ages[i] for i in valid_indices]
                else:
                    good_prev = np.zeros((0, 2), dtype=np.float32)
                    good_curr = np.zeros((0, 2), dtype=np.float32)
                    good_ages = []

                good_ages = [age + 1 for age in good_ages]

                if good_prev.shape[0] >= 5:
                    dt = timestamp - self.prev_timestamp
                    omega_rad, omega_deg, period = self.compute_angular_velocity(
                        good_prev, good_curr, dt, gray.shape
                    )
                    
                    rotation_increment = abs(omega_deg) * dt
                    self.cumulative_rotation += rotation_increment

                    # Store raw velocity
                    self.raw_velocities.append(omega_deg)
                    self.raw_timestamps.append(timestamp)

                    # Use moving median for smoothing
                    self.velocity_history.append(omega_deg)
                    smoothed_omega = float(np.median(list(self.velocity_history)))
                    smoothed_period = 360.0 / abs(smoothed_omega) if abs(smoothed_omega) > 1e-6 else float('inf')

                self.prev_points = good_curr.copy()
                self.point_ages = good_ages.copy()
        else:
            good_prev = np.zeros((0, 2), dtype=np.float32)
            good_curr = np.zeros((0, 2), dtype=np.float32)
            good_ages = []

        self.frames_since_refresh += 1
        need_refresh = (self.frames_since_refresh >= REFRESH_EVERY_N_FRAMES or
                        self.prev_points.shape[0] < MIN_TRACKING_POINTS)
        if need_refresh:
            print(f"[Frame {self.frame_count}] Refreshing points: {self.prev_points.shape[0]} -> ", end="")
            self.prev_points, self.point_ages = self.add_new_points(gray, self.prev_points, self.point_ages)
            self.frames_since_refresh = 0
            print(f"{self.prev_points.shape[0]} points")

        cycle_detected = self.check_cycle_completion(gray, timestamp)

        ages_for_vis = self.point_ages.copy()
        if len(ages_for_vis) < self.prev_points.shape[0]:
            ages_for_vis += [0] * (self.prev_points.shape[0] - len(ages_for_vis))

        vis_frame = self.visualize_tracking(patch.copy(), self.prev_points, self.prev_points, ages_for_vis,
                                            smoothed_omega, smoothed_period, cycle_detected)

        self.prev_gray = gray.copy()
        self.prev_timestamp = timestamp
        self.frame_count += 1

        return smoothed_omega, smoothed_period, vis_frame, cycle_detected


def main():
    tracker = AngularVelocityTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0 or np.isnan(fps):
        fps = 30.0
    print(f"Input FPS (reported): {fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    print("Starting smoothed angular velocity tracking...")
    print("Press 'q' to quit")

    start_time = time.time()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot fetch frame")
                break

            if VIDEO_PATH == 0 or str(VIDEO_PATH).isdigit():
                current_time = time.time() - start_time
            else:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time = pos_msec / 1000.0 if pos_msec is not None and pos_msec > 0 else (time.time() - start_time)

            frame_start = time.time()
            omega_deg, period, vis_frame, cycle_detected = tracker.process_frame(frame, current_time)
            frame_process_time = time.time() - frame_start

            tracker.fps_history.append(1.0 / frame_process_time if frame_process_time > 0 else 0.0)
            avg_fps = float(np.mean(list(tracker.fps_history))) if len(tracker.fps_history) > 0 else 0.0

            tracker.frame_times.append(current_time)
            tracker.angular_velocities.append(omega_deg)
            tracker.rotation_history.append(tracker.cumulative_rotation)

            if vis_frame is None:
                vis_frame = frame.copy()
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, vis_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if out is None and OUTPUT_VIDEO and vis_frame is not None:
                h, w = vis_frame.shape[:2]
                out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

            if out is not None and vis_frame is not None:
                out.write(vis_frame)

            if SHOW_VISUALIZATION and vis_frame is not None:
                cv2.imshow('Angular Velocity Tracker', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    # Enhanced plotting with additional smoothing
    if SAVE_PLOT and len(tracker.frame_times) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        times = np.array(tracker.frame_times)
        velocities = np.array(tracker.angular_velocities)
        rotations = np.array(tracker.rotation_history)

        # Apply Savitzky-Golay filter for even smoother curves
        if len(velocities) > 51:
            window_length = 51  # Must be odd
            polyorder = 3
            velocities_smooth = savgol_filter(velocities, window_length, polyorder)
        else:
            velocities_smooth = velocities

        ax1.plot(times, velocities, linewidth=0.5, alpha=0.3, label='Raw', color='lightblue')
        ax1.plot(times, velocities_smooth, linewidth=2, label='Smoothed', color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (deg/s)')
        ax1.set_title('Real-time Angular Velocity Tracking (Smoothed)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for cycle_time, duration in tracker.cycle_times:
            ax1.axvline(cycle_time, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.text(cycle_time, ax1.get_ylim()[1] * 0.9, f'{duration:.1f}s', rotation=90, va='top')

        periods = np.array([360.0 / abs(v) if abs(v) > 0.001 else np.nan for v in velocities_smooth])
        ax2.plot(times, periods, linewidth=2, color='green')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Period (s/rotation)')
        ax2.set_title('Rotation Period Over Time')
        ax2.set_ylim([0, min(50, np.nanmax(periods) if not np.isnan(np.nanmax(periods)) else 50)])
        ax2.grid(True, alpha=0.3)

        ax3.plot(times, rotations, linewidth=2, color='purple')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Cumulative Rotation (degrees)')
        ax3.set_title('Cumulative Rotation (resets after each cycle)')
        ax3.grid(True, alpha=0.3)

        for cycle_time, _ in tracker.cycle_times:
            ax3.axvline(cycle_time, color='red', linestyle='--', alpha=0.5, linewidth=2)

        plt.tight_layout()
        plt.savefig('angular_velocity_analysis.png', dpi=150)
        print(f"\nPlot saved to angular_velocity_analysis.png")

        valid_velocities = velocities_smooth[np.abs(velocities_smooth) > 0.1]
        if len(valid_velocities) > 0:
            print(f"\nStatistics:")
            print(f"  Total cycles: {tracker.cycles_completed}")
            print(f"  Mean velocity: {np.mean(valid_velocities):.2f} deg/s")
            print(f"  Std deviation: {np.std(valid_velocities):.2f} deg/s")
            print(f"  Mean period: {360.0 / np.mean(np.abs(valid_velocities)):.2f} s/rot")

            if len(tracker.cycle_times) > 1:
                cycle_durations = [dur for _, dur in tracker.cycle_times[1:]]
                print(f"  Mean cycle: {np.mean(cycle_durations):.2f} s")
                print(f"  Cycle std dev: {np.std(cycle_durations):.2f} s")

    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()