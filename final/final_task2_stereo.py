import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, ransac

from scipy.interpolate import griddata

# task2 height from stereo pics framework
# need better center measurement strategy
# apply:case_all_circle


class StereoHeightEstimator:
    """
    Estimates the height of an Aerial Object (AO) above ground using stereo vision.
    Uses triangulation with known baseline distance between two cameras.
    """
    
    def __init__(self, baseline_cm=120, focal_length_px=None):
        """
        Initialize the stereo height estimator.
        
        Args:
            baseline_cm: Distance between the two cameras in centimeters (default: 120cm)
            focal_length_px: Focal length in pixels (can be estimated from images)
        """
        self.baseline_cm = baseline_cm
        self.focal_length_px = focal_length_px
        
    def detect_ao_center(self, image, center_weight=0.2, auto_scale_margin=20, use_connected_components=True):
        """
        Enhanced version using connected components with automatic radius adjustment
        
        Parameters:
        -----------
        image : Input image
        center_weight : float
            Weight for center penalty (0 to 1). Higher values favor more central circles.
            0 = no penalty, 1 = maximum penalty for off-center circles
        auto_scale_margin : int
            Pixel margin to add beyond the farthest component point (default: 5 pixels).
            Ensures all edges are included with a small safety buffer.
        use_connected_components : bool
            If True, use connected components analysis to keep all central data points
        
        Returns:
        --------
        circle_mask : numpy array     Binary mask of the detected circle
        edges : numpy array           Edge detection result (for visualization)
        """

        H, W = image.shape[:2]
        img_center_x = W / 2
        img_center_y = H / 2
        max_distance = np.sqrt(img_center_x**2 + img_center_y**2)  # diagonal distance
        max_possible_radius = min(H, W) / 2  # Maximum reasonable radius

        # Color threshold, ocean only -- pick blue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([85, 40, 40])
        upper_blue = np.array([135, 255, 255])
        ocean_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Smooth mask (remove land + noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        if use_connected_components:
            print("  Using connected components analysis...")
            
            # First, create a center region mask to focus analysis
            # Only analyze the central 80% of the image to avoid background
            center_roi = np.zeros_like(ocean_mask)  # region of interest
            roi_margin_h = int(H * 0.1)  # 10% margin on top/bottom
            roi_margin_w = int(W * 0.1)  # 10% margin on left/right
            center_roi[roi_margin_h:H-roi_margin_h, roi_margin_w:W-roi_margin_w] = 255
            
            # Apply ROI constraint to ocean mask
            ocean_mask_roi = cv2.bitwise_and(ocean_mask, center_roi)
            
            # Find all connected components in the ROI-constrained ocean mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ocean_mask_roi, connectivity=8)
            
            # Filter components: keep those near the image center
            # This identifies the main globe region(s)
            central_components = []
            for i in range(1, num_labels):  # Skip background (label 0)
                cx, cy = centroids[i]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Distance from image center
                dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                
                # Keep components that are:
                # 1. Reasonably large (not tiny noise)
                # 2. Near the center of the image (stricter than before)
                if area > 200 and dist < max_distance * 0.5:  # Within 50% of diagonal
                    central_components.append(i)
            
            # print(f"  Found {len(central_components)} central components out of {num_labels-1} total")
            
            # Create mask of all central components
            central_mask = np.zeros_like(ocean_mask)
            for comp_id in central_components:
                central_mask[labels == comp_id] = 255
            
            # Extract ALL boundary points from the central components
            # Use all pixels on the edge, not just Canny-detected edges
            kernel_boundary = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(central_mask, kernel_boundary, iterations=1)
            boundary = cv2.subtract(central_mask, eroded)
            
            edges = boundary
            
            # Use boundary points for circle fitting
            boundary_ys, boundary_xs = np.where(boundary > 0)
            
            print(f"  Total central region pixels: {np.sum(central_mask > 0)}")
            print(f"  Boundary points: {len(boundary_xs)}")
            
            # Use boundary points
            if len(boundary_xs) > 100:
                points_xs = boundary_xs
                points_ys = boundary_ys
            else:
                # Fallback: use all central points
                ys, xs = np.where(central_mask > 0)
                points_xs = xs
                points_ys = ys
                
        else:
            # Apply Canny edge detection on the color mask (original method)
            edges = cv2.Canny(ocean_mask, 80, 180)
            points_ys, points_xs = np.where(edges > 0)
        
        # Extract edge points for circle fitting
        points = np.column_stack([points_xs, points_ys])
        print(f"  Using {len(points)} points for circle fitting")

        # RANSAC circle fitting with multiple trials to find LARGEST circle that contains the data
        model = CircleModel()
        best_circle = None
        best_score = -np.inf
        n_iterations = 10  # More iterations for better results
        
        if use_connected_components:
            for i in range(n_iterations):
                try:
                    circle, inliers = ransac(points, CircleModel, min_samples=3, 
                                            residual_threshold=3, max_trials=2000)
                    xc, yc, r = circle.params
                    
                    r=r+auto_scale_margin # connected component tend to underscale, extend radius a bit

                    # we want the LARGEST circle that contains most points
                    # use_connected_components:
                    # Calculate how many of our component points are inside this circle
                    all_ys, all_xs = np.where(central_mask > 0)
                    distances = np.sqrt((all_xs - xc)**2 + (all_ys - yc)**2)
                    points_inside = np.sum(distances <= r)
                    coverage_ratio = points_inside / len(all_xs) if len(all_xs) > 0 else 0
                    
                    # Calculate distance from image center
                    distance_from_center = np.sqrt((xc - img_center_x)**2 + (yc - img_center_y)**2)
                    normalized_distance = distance_from_center / max_distance
                    
                    # Calculate radius score (larger is better)
                    normalized_radius = r / max_possible_radius
                    
                    # Score prioritizes: good coverage, large radius, central position
                    center_score = 1.0 - normalized_distance
                    radius_score = normalized_radius
                    
                    # Heavily prioritize coverage and radius for better globe capture
                    combined_score = (0.4 * coverage_ratio + 0.4 * radius_score + 0.2 * center_score)
                    
                    print(f"  Trial {i+1}: center=({xc:.1f}, {yc:.1f}), r={r:.1f}, "
                            f"coverage={coverage_ratio:.3f}, radius_score={radius_score:.3f}, "
                            f"score={combined_score:.3f}")

                    if combined_score > best_score:
                        best_score = combined_score
                        best_circle = (xc, yc, r)
                    
                except Exception as e:
                    print(f"  Trial {i+1} failed: {e}")
                    continue
        
        else:
            best_circle, inliers = ransac(points, CircleModel, min_samples=3,
                                    residual_threshold=3, max_trials=2000)
            xc, yc, r = best_circle.params
            print(f"  Hough Circle: center=({xc:.1f}, {yc:.1f}), r={r:.1f} ")
                
        
        xc, yc, r = best_circle
        distance_from_center = np.sqrt((xc - img_center_x)**2 + (yc - img_center_y)**2)
        
        print(f"\n  BEST CIRCLE SELECTED:")
        print(f"    Center: ({xc:.1f}, {yc:.1f})")
        print(f"    Radius: {r:.1f}")
        print(f"    Image center: ({img_center_x:.1f}, {img_center_y:.1f})")
        print(f"    Distance from image center: {distance_from_center:.1f} pixels")


        # Build final circle mask
        Y, X = np.ogrid[:H, :W]
        circle_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        center=np.array([xc,yc])
        return circle_mask, center, r ,edges

    def match_features_sift(self, img_left, img_right):
        """
        Match features between left and right images using SIFT.
        Returns matched keypoints and their coordinates.
        """
        # Convert to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp_left, des_left = sift.detectAndCompute(gray_left, None)
        kp_right, des_right = sift.detectAndCompute(gray_right, None)
        
        print(f"  Detected {len(kp_left)} keypoints in left image")
        print(f"  Detected {len(kp_right)} keypoints in right image")
        
        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des_left, des_right, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        pts_left = []
        pts_right = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    pts_left.append(kp_left[m.queryIdx].pt)
                    pts_right.append(kp_right[m.trainIdx].pt)
        
        print(f"  Found {len(good_matches)} good matches")
        
        return np.array(pts_left), np.array(pts_right), good_matches, kp_left, kp_right
    
    def estimate_focal_length(self, img_shape):
        """
        Estimate focal length based on typical camera parameters.
        Uses a common assumption: focal_length ≈ image_width
        """
        if self.focal_length_px is None:
            # Typical assumption for consumer cameras
            self.focal_length_px = img_shape[1]  # Use image width
            print(f"  Estimated focal length: {self.focal_length_px:.1f} pixels")
        return self.focal_length_px
    
    def calculate_disparity(self, pts_left, pts_right):
        """
        Calculate horizontal disparity between matched points.
        Disparity = x_left - x_right (for parallel stereo setup)
        """
        disparities = pts_left[:, 0] - pts_right[:, 0]
        return disparities
    
    def disparity_to_depth(self, disparity, focal_length, baseline):
        """
        Convert disparity to depth using triangulation formula:
        Depth (Z) = (focal_length * baseline) / disparity
        
        Args:
            disparity: Horizontal pixel disparity
            focal_length: Focal length in pixels
            baseline: Distance between cameras in cm
            
        Returns:
            Depth in centimeters
        """
        # Avoid division by zero
        valid_mask = disparity > 1e-6
        depth = np.zeros_like(disparity)
        depth[valid_mask] = (focal_length * baseline) / disparity[valid_mask]
        return depth
    
    def estimate_height_from_stereo(self, img_left, img_right, 
                                   ground_reference_y=None, visualize=True):
        """
        Main method to estimate AO height above ground using stereo vision.
        
        Args:
            img_left: Left camera image
            img_right: Right camera image
            ground_reference_y: Y-coordinate of ground plane in image (if None, uses bottom)
            visualize: Whether to show visualization
            
        Returns:
            Dictionary with height estimate and related measurements
        """
        print("\n" + "="*70)
        print("STEREO VISION HEIGHT ESTIMATION")
        print("="*70)
        
        # Step 1: Detect AO in both images
        print("\n[1] Detecting AO in left image...")
        mask_left, center_left, radius_left ,_ = self.detect_ao_center(img_left)
        
        print("\n[2] Detecting AO in right image...")
        mask_right, center_right, radius_right ,_ = self.detect_ao_center(img_right)
        
        if center_left is None or center_right is None:
            print("ERROR: Failed to detect AO in both images")
            return None
        
        # Step 2: Match features
        print("\n[3] Matching features between images...")
        pts_left, pts_right, matches, kp_left, kp_right = self.match_features_sift(
            img_left, img_right)
        
        if len(pts_left) == 0:
            print("ERROR: No feature matches found")
            return None
        
        # Step 3: Estimate focal length
        print("\n[4] Camera calibration...")
        focal_length = self.estimate_focal_length(img_left.shape)
        
        # Step 4: Calculate disparities and depths
        print("\n[5] Computing depth map...")
        disparities = self.calculate_disparity(pts_left, pts_right)
        depths = self.disparity_to_depth(disparities, focal_length, self.baseline_cm)
        
        # Filter out invalid depths
        valid_depths = depths[(depths > 0) & (depths < 10000)]
        
        if len(valid_depths) == 0:
            print("ERROR: No valid depth measurements")
            return None
        
        print(f"  Valid depth measurements: {len(valid_depths)}")
        print(f"  Depth range: {valid_depths.min():.1f} - {valid_depths.max():.1f} cm")
        print(f"  Mean depth: {valid_depths.mean():.1f} cm")
        print(f"  Median depth: {np.median(valid_depths):.1f} cm")
        
        # Step 5: Calculate AO center disparity
        print("\n[6] Calculating AO center depth...")
        ao_disparity = center_left[0] - center_right[0]
        ao_depth = self.disparity_to_depth(ao_disparity, focal_length, self.baseline_cm)
        
        print(f"  AO center disparity: {ao_disparity:.2f} pixels")
        print(f"  AO depth from camera: {ao_depth:.1f} cm")
        
        # Step 6: Calculate height above ground
        print("\n[7] Computing height above ground...")
        
        # Determine ground reference
        if ground_reference_y is None:
            ground_reference_y = img_left.shape[0]  # Bottom of image
        
        # Calculate vertical pixel distance from AO bottom to ground
        ao_bottom_y = center_left[1] + radius_left
        vertical_pixels = ground_reference_y - ao_bottom_y
        
        # Convert pixel height to real-world height using similar triangles
        # Real height = (pixel_height / focal_length) * depth
        height_cm = (vertical_pixels / focal_length) * ao_depth
        
        print(f"  AO bottom Y-coordinate: {ao_bottom_y:.1f} pixels")
        print(f"  Ground reference Y-coordinate: {ground_reference_y:.1f} pixels")
        print(f"  Vertical pixel distance: {vertical_pixels:.1f} pixels")
        print(f"  ESTIMATED HEIGHT: {height_cm:.1f} cm ({height_cm/100:.2f} meters)")
        
        # Step 7: Validation using multiple depth measurements
        print("\n[8] Validation...")
        
        # Find points near the AO
        ao_region_mask = (np.abs(pts_left[:, 0] - center_left[0]) < radius_left * 1.5) & \
                        (np.abs(pts_left[:, 1] - center_left[1]) < radius_left * 1.5)
        
        if np.sum(ao_region_mask) > 5:
            ao_depths = depths[ao_region_mask]
            ao_depths_valid = ao_depths[(ao_depths > 0) & (ao_depths < 10000)]
            
            if len(ao_depths_valid) > 0:
                validation_depth = np.median(ao_depths_valid)
                validation_height = (vertical_pixels / focal_length) * validation_depth
                
                print(f"  Validation depth (median of {len(ao_depths_valid)} points): {validation_depth:.1f} cm")
                print(f"  Validation height estimate: {validation_height:.1f} cm")
                print(f"  Difference: {abs(height_cm - validation_height):.1f} cm ({abs(height_cm - validation_height)/height_cm*100:.1f}%)")
            else:
                validation_depth = None
                validation_height = None
        else:
            validation_depth = None
            validation_height = None
        
        # Visualization
        if visualize:
            self.visualize_results(img_left, img_right, pts_left, pts_right,
                                  center_left, center_right, radius_left,
                                  matches, kp_left, kp_right, height_cm)
        
        # Return results
        results = {
            'height_cm': height_cm,
            'height_m': height_cm / 100,
            'ao_depth_cm': ao_depth,
            'ao_disparity_px': ao_disparity,
            'validation_height_cm': validation_height,
            'center_left': center_left,
            'center_right': center_right,
            'num_matches': len(matches),
            'focal_length_px': focal_length,
            'baseline_cm': self.baseline_cm
        }
        
        print("\n" + "="*70)
        print(f"FINAL RESULT: AO Height = {height_cm:.1f} cm ({height_cm/100:.2f} m)")
        print("="*70 + "\n")
        
        return results
    


    def visualize_results(self, img_left, img_right, pts_left, pts_right,
                                center_left, center_right, radius_left,
                                matches, kp_left, kp_right, height_cm):
        """
        Visualize the stereo matching and height estimation results with improved disparity map.
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Matched features
        ax1 = plt.subplot(2, 2, 1)
        img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                    matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax1.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        ax1.set_title('Feature Matches (showing 50)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Left image with AO detection
        ax2 = plt.subplot(2, 2, 2)
        overlay_left = img_left.copy()
        cv2.circle(overlay_left, tuple(center_left.astype(int)), int(radius_left), 
                (0, 255, 0), 3)
        cv2.circle(overlay_left, tuple(center_left.astype(int)), 5, (255, 0, 0), -1)
        ax2.imshow(cv2.cvtColor(overlay_left, cv2.COLOR_BGR2RGB))
        ax2.set_title('Left Image - AO Detection', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Right image with AO detection
        ax3 = plt.subplot(2, 2, 3)
        overlay_right = img_right.copy()
        cv2.circle(overlay_right, tuple(center_right.astype(int)), int(radius_left), 
                (0, 255, 0), 3)
        cv2.circle(overlay_right, tuple(center_right.astype(int)), 5, (255, 0, 0), -1)
        ax3.imshow(cv2.cvtColor(overlay_right, cv2.COLOR_BGR2RGB))
        ax3.set_title('Right Image - AO Detection', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. IMPROVED Disparity visualization
        ax4 = plt.subplot(2, 2, 4)
        
        # Calculate disparities for all matched points
        disparities = pts_left[:, 0] - pts_right[:, 0]
        
        # Filter valid disparities (remove outliers)
        valid_mask = (disparities > 0) & (disparities < 200)  # Adjust threshold as needed
        valid_pts = pts_left[valid_mask]
        valid_disp = disparities[valid_mask]
        
        if len(valid_disp) > 10:
            # Create interpolated disparity map
            h, w = img_left.shape[:2]
            
            # Create grid for interpolation
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Interpolate disparity values across the image
            # Using 'nearest' for speed, or 'linear'/'cubic' for smoother results
            disparity_map = griddata(
                points=valid_pts,
                values=valid_disp,
                xi=(grid_x, grid_y),
                method='nearest',  # Try 'linear' or 'cubic' for smoother results
                fill_value=0
            )
            
            # Optional: Apply Gaussian blur for smoother visualization
            disparity_map = cv2.GaussianBlur(disparity_map.astype(np.float32), (15, 15), 0)
            
            # Display with better colormap and range
            vmin, vmax = np.percentile(valid_disp, [5, 95])  # Use percentiles to avoid outliers
            im = ax4.imshow(disparity_map, cmap='jet', vmin=vmin, vmax=vmax)
            
            # Overlay matched points
            ax4.scatter(valid_pts[:, 0], valid_pts[:, 1], c='white', s=1, alpha=0.3)
            
        else:
            # Fallback: show sparse disparity
            disparity_map = np.zeros(img_left.shape[:2])
            for pt, disp in zip(valid_pts, valid_disp):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    disparity_map[y, x] = disp
            im = ax4.imshow(disparity_map, cmap='jet')
        
        ax4.set_title(f'Disparity Map (Interpolated)\nEstimated Height: {height_cm:.1f} cm', 
                    fontsize=12, fontweight='bold')
        ax4.axis('off')
        cbar = plt.colorbar(im, ax=ax4, label='Disparity (pixels)')
        
        # Add depth scale annotation
        if len(valid_disp) > 0:
            focal_length = self.focal_length_px
            baseline = self.baseline_cm
            depth_min = (focal_length * baseline) / vmax if vmax > 0 else 0
            depth_max = (focal_length * baseline) / vmin if vmin > 0 else 0
            ax4.text(0.02, 0.98, f'Depth: {depth_min:.0f}-{depth_max:.0f} cm', 
                    transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('stereo_height_estimation.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to: stereo_height_estimation.png")
        plt.show()


# Alternative: Dense stereo matching using OpenCV's StereoSGBM
def create_dense_disparity_map(img_left, img_right):
    """
    Create a dense disparity map using Semi-Global Block Matching.
    This provides disparity values for most pixels, not just feature matches.
    """
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Create StereoSGBM object
    window_size = 5
    min_disp = 0
    num_disp = 16 * 10  # Must be divisible by 16
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Filter invalid disparities
    disparity[disparity <= 0] = np.nan
    
    return disparity


# Example usage in visualization:
def visualize_with_dense_disparity(self, img_left, img_right, pts_left, pts_right,
                                center_left, center_right, radius_left,
                                matches, kp_left, kp_right, height_cm):
    """
    Visualization using dense stereo matching for better disparity map.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Subplots 1-3 same as before...
    ax1 = plt.subplot(2, 2, 1)
    img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax1.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    ax1.set_title('Feature Matches', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    overlay_left = img_left.copy()
    cv2.circle(overlay_left, tuple(center_left.astype(int)), int(radius_left), 
            (0, 255, 0), 3)
    ax2.imshow(cv2.cvtColor(overlay_left, cv2.COLOR_BGR2RGB))
    ax2.set_title('Left Image', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    overlay_right = img_right.copy()
    cv2.circle(overlay_right, tuple(center_right.astype(int)), int(radius_left), 
            (0, 255, 0), 3)
    ax3.imshow(cv2.cvtColor(overlay_right, cv2.COLOR_BGR2RGB))
    ax3.set_title('Right Image', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Dense disparity map
    ax4 = plt.subplot(2, 2, 4)
    print("  Computing dense disparity map...")
    disparity_map = create_dense_disparity_map(img_left, img_right)
    
    im = ax4.imshow(disparity_map, cmap='jet')
    ax4.set_title(f'Dense Disparity Map\nEstimated Height: {height_cm:.1f} cm', 
                fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, label='Disparity (pixels)')
    
    plt.tight_layout()
    plt.savefig('stereo_height_estimation.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: stereo_height_estimation.png")
    plt.show()

# USAGE EXAMPLE
if __name__ == "__main__":
    # Initialize the estimator with known baseline
    estimator = StereoHeightEstimator(baseline_cm=120)
    
    # Load stereo pair images
    img_left = cv2.imread('left_image.png')  # Replace with your left image path
    img_right = cv2.imread('right_image.png')  # Replace with your right image path
    
    if img_left is None or img_right is None:
        print("ERROR: Could not load images. Please check file paths.")
        print("\nUsage:")
        print("  1. Place your left and right stereo images in the same directory")
        print("  2. Update the file paths in the code")
        print("  3. Run the script")
    else:
        # Estimate height
        results = estimator.estimate_height_from_stereo(
            img_left, 
            img_right,
            ground_reference_y=None,  # Uses bottom of image
            visualize=True
        )
        
        if results:
            print("\n" + "="*70)
            print("SUMMARY OF RESULTS")
            print("="*70)
            print(f"Height Above Ground: {results['height_cm']:.1f} cm ({results['height_m']:.2f} m)")
            print(f"AO Depth from Camera: {results['ao_depth_cm']:.1f} cm")
            print(f"AO Disparity: {results['ao_disparity_px']:.2f} pixels")
            print(f"Number of Feature Matches: {results['num_matches']}")
            print(f"Focal Length: {results['focal_length_px']:.1f} pixels")
            print(f"Camera Baseline: {results['baseline_cm']} cm")
            if results['validation_height_cm']:
                print(f"Validation Height: {results['validation_height_cm']:.1f} cm")
            print("="*70)


# Initialize with your baseline
estimator = StereoHeightEstimator(baseline_cm=120)

# Load your stereo pair
img_left = cv2.imread('task2_stereo/left_camera.jpg')
img_right = cv2.imread('task2_stereo/right_camera.jpg')

# Estimate height with validation
results = estimator.estimate_height_from_stereo(img_left, img_right) # true=20.4m

print(f"Height: {results['height_cm']:.1f} cm")