import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, ransac
import os

# find center from video framework
# handle outlier
# need better center measurement strategy

def color_then_circle(image):
    """Extract AO mask and find center using color segmentation + shape fitting"""
    
    # Color threshold, ocean only
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([135, 255, 255])
    ocean_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Smooth mask (remove land + noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply Canny edge detection on the color mask
    edges = cv2.Canny(ocean_mask, 50, 150)
    
    # Extract edge points
    ys, xs = np.where(edges > 0)
    
    if len(xs) < 20:
        print("Insufficient edge points, fallback to color mask only")
        return ocean_mask, None
    
    points = np.column_stack([xs, ys])

    # RANSAC circle fitting on edge points
    model = CircleModel()
    try:
        circle, inliers = ransac(points, CircleModel, min_samples=3, 
                                residual_threshold=2, max_trials=2000)
        xc, yc, r = circle.params
        circle_center = [xc, yc]
    except Exception as e:
        print(f"Circle fit failed: {e}, fallback to mask")
        return ocean_mask, None
        
    try:
        # Calculate circle fitting error
        distances_circle = np.abs(np.sqrt((xs - xc)**2 + (ys - yc)**2) - r)
        circle_error = np.mean(distances_circle)
    except np.linalg.LinAlgError:
        circle_error = float('inf')

    # FIT ELLIPSE 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found")
        return np.zeros_like(ocean_mask), None
    
    # Choose the better fit
    H, W = ocean_mask.shape
    final_mask = np.zeros_like(ocean_mask)
    
    print(f"Frame: Circle fit (error: {circle_error:.2f})")
    Y, X = np.ogrid[:H, :W]
    final_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
    center = circle_center
    
    return final_mask, center


def analyze_video_center_shift(video_path, output_dir='results/center_tracking'):
    """Process video and track AO center across frames"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*70}")
    print(f"Video Analysis: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"{'='*70}\n")
    
    # Storage for tracking data
    centers = []
    timestamps = []
    frame_numbers = []
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx + 1}/{total_frames}...", end=' ')
        
        # Find center using the provided method
        mask, center = color_then_circle(frame)
        
        if center is not None:
            centers.append(center)
            timestamps.append(frame_idx / fps)
            frame_numbers.append(frame_idx)
            print(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
        else:
            print("Center not found")
        
        frame_idx += 1
    
    cap.release()
    
    # Convert to numpy arrays
    centers = np.array(centers)
    timestamps = np.array(timestamps)
    
    print(f"\n{'='*70}")
    print(f"Tracking Complete!")
    print(f"Successfully tracked {len(centers)}/{total_frames} frames")
    print(f"{'='*70}\n")
    
    # Analysis
    if len(centers) > 0:
        analyze_center_movement(centers, timestamps, frame_numbers, output_dir, 
                               video_path, width, height)
    else:
        print("No centers detected in any frame!")
    
    return centers, timestamps


def remove_outliers(centers, timestamps, frame_numbers, method='iqr', threshold=3.0):
    """Remove outlier points using IQR or Z-score method"""
    
    if len(centers) < 10:
        print("Too few points for outlier removal")
        return centers, timestamps, frame_numbers, []
    
    # Calculate center-to-center distances (movement between consecutive frames)
    distances = np.sqrt(np.sum(np.diff(centers, axis=0)**2, axis=1))
    
    # Pad with 0 for first frame
    distances = np.concatenate([[0], distances])
    
    if method == 'iqr':
        # Interquartile Range method
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Identify inliers
        inliers = (distances >= lower_bound) & (distances <= upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        z_scores = np.abs((distances - mean_dist) / std_dist)
        inliers = z_scores < threshold
    
    else:
        # MAD (Median Absolute Deviation) method - more robust
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        modified_z_scores = 0.6745 * (distances - median_dist) / mad
        inliers = np.abs(modified_z_scores) < threshold
    
    outliers_removed = np.sum(~inliers)
    outlier_indices = np.where(~inliers)[0]
    
    # Convert frame_numbers to numpy array if it's a list
    frame_numbers_array = np.array(frame_numbers)
    
    print(f"\n{'='*70}")
    print(f"OUTLIER DETECTION AND REMOVAL")
    print(f"{'='*70}")
    print(f"Method: {method.upper()}")
    print(f"Total points: {len(centers)}")
    print(f"Outliers detected: {outliers_removed}")
    if outliers_removed > 0:
        outlier_frames = frame_numbers_array[~inliers].tolist()
        print(f"Outlier frames: {outlier_frames}")
    else:
        print(f"Outlier frames: None")
    print(f"Points remaining: {np.sum(inliers)}")
    print(f"{'='*70}\n")
    
    # Return cleaned data
    return centers[inliers], timestamps[inliers], frame_numbers_array[inliers], outlier_indices


def analyze_center_movement(centers, timestamps, frame_numbers, output_dir, 
                           video_path, width, height):
    """Analyze and visualize center movement"""
    
    # Remove outliers before analysis
    centers_clean, timestamps_clean, frames_clean, outlier_idx = remove_outliers(
        centers, timestamps, frame_numbers, method='mad', threshold=3.5
    )
    
    # Use cleaned data for all analysis
    centers = centers_clean
    timestamps = timestamps_clean
    frame_numbers = frames_clean
    
    # Calculate statistics
    mean_center = np.mean(centers, axis=0)
    std_center = np.std(centers, axis=0)
    
    # Calculate displacements from mean
    displacements = centers - mean_center
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    
    max_displacement = np.max(distances)
    mean_displacement = np.mean(distances)
    
    # Calculate velocity (pixel/second)
    if len(centers) > 1:
        dt = np.diff(timestamps)
        dx = np.diff(centers[:, 0])
        dy = np.diff(centers[:, 1])
        velocities = np.sqrt(dx**2 + dy**2) / dt
        mean_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
    else:
        velocities = None
        mean_velocity = 0
        max_velocity = 0
    
    # Print statistics
    print("\n" + "="*70)
    print("CENTER SHIFT ANALYSIS (AFTER OUTLIER REMOVAL)")
    print("="*70)
    print(f"Frames analyzed: {len(centers)}")
    print(f"Mean Center Position: ({mean_center[0]:.2f}, {mean_center[1]:.2f})")
    print(f"Standard Deviation: X={std_center[0]:.2f}px, Y={std_center[1]:.2f}px")
    print(f"Maximum Displacement: {max_displacement:.2f} pixels")
    print(f"Mean Displacement: {mean_displacement:.2f} pixels")
    print(f"Mean Velocity: {mean_velocity:.2f} pixels/sec")
    print(f"Max Velocity: {max_velocity:.2f} pixels/sec")
    print("="*70 + "\n")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Trajectory plot (spatial)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(centers[:, 0], centers[:, 1], 'b-', alpha=0.5, linewidth=1)
    ax1.scatter(centers[:, 0], centers[:, 1], c=timestamps, cmap='viridis', 
                s=50, edgecolors='black', linewidth=0.5)
    ax1.plot(mean_center[0], mean_center[1], 'r*', markersize=20, 
             label=f'Mean Center\n({mean_center[0]:.1f}, {mean_center[1]:.1f})')
    ax1.set_xlabel('X Position (pixels)', fontsize=11)
    ax1.set_ylabel('Y Position (pixels)', fontsize=11)
    ax1.set_title('AO Center Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('Time (seconds)', fontsize=10)
    
    # 2. X position over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(timestamps, centers[:, 0], 'b.-', linewidth=1.5, markersize=4)
    ax2.axhline(mean_center[0], color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_center[0]:.2f}px')
    ax2.fill_between(timestamps, 
                     mean_center[0] - std_center[0], 
                     mean_center[0] + std_center[0], 
                     alpha=0.3, color='red', label=f'±1σ: {std_center[0]:.2f}px')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('X Position (pixels)', fontsize=11)
    ax2.set_title('Horizontal Movement', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Y position over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(timestamps, centers[:, 1], 'g.-', linewidth=1.5, markersize=4)
    ax3.axhline(mean_center[1], color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_center[1]:.2f}px')
    ax3.fill_between(timestamps, 
                     mean_center[1] - std_center[1], 
                     mean_center[1] + std_center[1], 
                     alpha=0.3, color='red', label=f'±1σ: {std_center[1]:.2f}px')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Y Position (pixels)', fontsize=11)
    ax3.set_title('Vertical Movement', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Displacement from mean
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(timestamps, distances, 'purple', linewidth=2)
    ax4.axhline(mean_displacement, color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {mean_displacement:.2f}px')
    ax4.axhline(max_displacement, color='red', linestyle='--', linewidth=2,
                label=f'Max: {max_displacement:.2f}px')
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Distance from Mean (pixels)', fontsize=11)
    ax4.set_title('Displacement Magnitude', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Velocity over time
    if velocities is not None:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(timestamps[1:], velocities, 'orange', linewidth=1.5)
        ax5.axhline(mean_velocity, color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_velocity:.2f}px/s')
        ax5.set_xlabel('Time (seconds)', fontsize=11)
        ax5.set_ylabel('Velocity (pixels/sec)', fontsize=11)
        ax5.set_title('Center Movement Velocity', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # 6. Distribution histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(distances, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax6.axvline(mean_displacement, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_displacement:.2f}px')
    ax6.axvline(max_displacement, color='orange', linestyle='--', linewidth=2,
                label=f'Max: {max_displacement:.2f}px')
    ax6.set_xlabel('Displacement (pixels)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Displacement Distribution', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'center_tracking_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {output_path}")
    plt.show()
    
    # Save data to CSV
    csv_path = os.path.join(output_dir, 'center_tracking_data.csv')
    with open(csv_path, 'w') as f:
        f.write("Frame,Time(s),X(px),Y(px),Displacement(px)\n")
        for i, (frame, time, center, dist) in enumerate(zip(frame_numbers, timestamps, centers, distances)):
            f.write(f"{frame},{time:.3f},{center[0]:.2f},{center[1]:.2f},{dist:.2f}\n")
    print(f"Tracking data saved to: {csv_path}")


# MAIN EXECUTION
if __name__ == "__main__":
    video_path = "center_shift.avi"
    
    if os.path.exists(video_path):
        centers, timestamps = analyze_video_center_shift(video_path)
    else:
        print(f"Error: Video file '{video_path}' not found!")
        print("Please ensure the video file is in the same directory as this script.")