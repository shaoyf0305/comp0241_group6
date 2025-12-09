import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel
from skimage.measure import ransac
import glob

#circle and ellipse--all test cases
# restriction: 1. not for small objects   2. rely on picture quality
#designed for task1--optimal
# better at handle noise

# use circle model for circles, use contour for ellipse

# METHOD 3: color, then choose between circle/ellipse

def color_then_circle_or_ellipse(image, center_weight=0.2, auto_scale_margin=20, use_connected_components=True):
    """
    Enhanced version using connected components with both circle and ellipse fitting
    Compares both methods and chooses the one with better coverage
    
    Parameters:
    -----------
    image : Input image
    center_weight : float
        Weight for center penalty (0 to 1). Higher values favor more central shapes.
    auto_scale_margin : int
        Pixel margin to add beyond the farthest component point (default: 20 pixels).
    use_connected_components : bool
        If True, use connected components analysis to keep all central data points
    
    Returns:
    --------
    final_mask : numpy array       Binary mask of the detected shape
    edges : numpy array             Edge detection result (for visualization)
    shape_type : str                'circle' or 'ellipse' indicating which was chosen
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
            # 2. Near the center of the image
            if area > 200 and dist < max_distance * 0.5:  # Within 50% of diagonal
                central_components.append(i)
        
        # Create mask of all central components
        central_mask = np.zeros_like(ocean_mask)
        for comp_id in central_components:
            central_mask[labels == comp_id] = 255
        
        # Extract boundary points from the central components
        kernel_boundary = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(central_mask, kernel_boundary, iterations=1)
        boundary = cv2.subtract(central_mask, eroded)
        
        edges = boundary
        
        # Use boundary points for shape fitting
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
        
        # For non-connected components, use all edge points
        ys, xs = np.where(edges > 0)
        central_mask = ocean_mask
    
    # Extract points for shape fitting
    points = np.column_stack([points_xs, points_ys])
    print(f"  Using {len(points)} points for shape fitting")
    
    # Get all interest points for coverage calculation
    all_ys, all_xs = np.where(central_mask > 0)
    total_interest_points = len(all_xs)
    print(f"  Total interest points to cover: {total_interest_points}")

    # ========== FIT CIRCLE ==========
    print("\n  === FITTING CIRCLE ===")
    best_circle = None
    best_circle_coverage = 0
    n_iterations = 10
    
    if use_connected_components and len(points) >= 3:
        for i in range(n_iterations):
            try:
                circle, inliers = ransac(points, CircleModel, min_samples=3, 
                                        residual_threshold=3, max_trials=2000)
                xc, yc, r = circle.params
                
                # Check if center is within frame boundary
                if not (0 <= xc < W and 0 <= yc < H):
                    print(f"  Trial {i+1}: Circle center outside frame, skipping")
                    continue
                
                r = r + auto_scale_margin  # Extend radius a bit
                
                # Calculate coverage: how many interest points are inside this circle
                distances = np.sqrt((all_xs - xc)**2 + (all_ys - yc)**2)
                points_inside = np.sum(distances <= r)
                coverage_ratio = points_inside / total_interest_points if total_interest_points > 0 else 0
                
                print(f"  Trial {i+1}: center=({xc:.1f}, {yc:.1f}), r={r:.1f}, "
                      f"coverage={points_inside}/{total_interest_points} ({coverage_ratio:.3f})")

                if coverage_ratio > best_circle_coverage:
                    best_circle_coverage = coverage_ratio
                    best_circle = (xc, yc, r)
                
            except Exception as e:
                print(f"  Trial {i+1} failed: {e}")
                continue
    else:
        try:
            circle, inliers = ransac(points, CircleModel, min_samples=3,
                                  residual_threshold=3, max_trials=2000)
            xc, yc, r = circle.params
            
            # Check if center is within frame boundary
            if 0 <= xc < W and 0 <= yc < H:
                distances = np.sqrt((all_xs - xc)**2 + (all_ys - yc)**2)
                points_inside = np.sum(distances <= r)
                best_circle_coverage = points_inside / total_interest_points if total_interest_points > 0 else 0
                best_circle = (xc, yc, r)
                print(f"  Circle: center=({xc:.1f}, {yc:.1f}), r={r:.1f}, coverage={best_circle_coverage:.3f}")
            else:
                print(f"  Circle center outside frame, skipping")
        except Exception as e:
            print(f"  Circle fitting failed: {e}")

    # ========== FIT ELLIPSE ==========
    print("\n  === FITTING ELLIPSE ===")
    best_ellipse = None
    best_ellipse_coverage = 0
    
    if len(points) >= 5:
        for i in range(n_iterations):
            try:
                # Sample points for ellipse fitting
                if len(points) > 1000:
                    # Subsample for efficiency
                    indices = np.random.choice(len(points), 1000, replace=False)
                    sample_points = points[indices]
                else:
                    sample_points = points
                
                # Fit ellipse using OpenCV
                contour = sample_points.reshape(-1, 1, 2).astype(np.int32)
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (MA, ma), angle = ellipse
                
                # Check if center is within frame boundary
                if not (0 <= cx < W and 0 <= cy < H):
                    print(f"  Trial {i+1}: Ellipse center outside frame, skipping")
                    continue
                
                # Add margin to axes
                MA = MA + 2 * auto_scale_margin
                ma = ma + 2 * auto_scale_margin
                
                # Semi-axes
                a = MA / 2
                b = ma / 2
                angle_rad = np.deg2rad(angle)
                
                # Calculate coverage: transform points to ellipse coordinate system
                xp = all_xs - cx
                yp = all_ys - cy
                xr = xp * np.cos(angle_rad) + yp * np.sin(angle_rad)
                yr = -xp * np.sin(angle_rad) + yp * np.cos(angle_rad)
                
                # Check if points are inside ellipse
                inside = (xr**2 / a**2 + yr**2 / b**2) <= 1
                points_inside = np.sum(inside)
                coverage_ratio = points_inside / total_interest_points if total_interest_points > 0 else 0
                
                print(f"  Trial {i+1}: center=({cx:.1f}, {cy:.1f}), axes=({a:.1f}, {b:.1f}), "
                      f"angle={angle:.1f}°, coverage={points_inside}/{total_interest_points} ({coverage_ratio:.3f})")

                if coverage_ratio > best_ellipse_coverage:
                    best_ellipse_coverage = coverage_ratio
                    best_ellipse = ((cx, cy), (MA, ma), angle)
                
            except Exception as e:
                print(f"  Trial {i+1} failed: {e}")
                continue
    else:
        print("  Insufficient points for ellipse fitting")

    # ========== COMPARE AND CHOOSE ==========
    print("\n  === COMPARISON ===")
    
    # Calculate scores: high coverage + small size = better score
    # Score = coverage / normalized_size
    # This rewards tight-fitting shapes that capture the data efficiently
    
    circle_score = 0
    ellipse_score = 0
    
    if best_circle is not None:
        xc, yc, r = best_circle
        # Normalize radius by max possible radius
        normalized_circle_size = r / max_possible_radius
        # Score: coverage divided by size (higher coverage and smaller size = better)
        circle_score = best_circle_coverage / (normalized_circle_size + 0.3)  # +0.1 to avoid division by zero
        print(f"  Circle: coverage={best_circle_coverage:.3f}, radius={r:.1f}, "
              f"norm_size={normalized_circle_size:.3f}, score={circle_score:.3f}")
    else:
        print(f"  Circle: No valid fit")
    
    if best_ellipse is not None:
        (cx, cy), (MA, ma), angle = best_ellipse
        # Normalize ellipse size by average of semi-axes
        avg_axis = (MA/2 + ma/2) / 2
        normalized_ellipse_size = avg_axis / max_possible_radius
        # Score: coverage divided by size
        ellipse_score = best_ellipse_coverage / (normalized_ellipse_size + 0.3)
        print(f"  Ellipse: coverage={best_ellipse_coverage:.3f}, avg_axis={avg_axis:.1f}, "
              f"norm_size={normalized_ellipse_size:.3f}, score={ellipse_score:.3f}")
    else:
        print(f"  Ellipse: No valid fit")
    
    final_mask = np.zeros((H, W), dtype=np.uint8)
    shape_type = None
    
    if best_circle is None and best_ellipse is None:
        print("  No valid shapes found, returning ocean mask")
        return ocean_mask, edges, "none"
    elif best_circle is None:
        print("  → Chose ELLIPSE (only valid option)")
        (cx, cy), (MA, ma), angle = best_ellipse
        cv2.ellipse(final_mask, best_ellipse, 255, -1)
        shape_type = "ellipse"
        print(f"  Final: center=({cx:.1f}, {cy:.1f}), axes=({MA/2:.1f}, {ma/2:.1f}), angle={angle:.1f}°")
    elif best_ellipse is None:
        print("  → Chose CIRCLE (only valid option)")
        xc, yc, r = best_circle
        Y, X = np.ogrid[:H, :W]
        final_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        shape_type = "circle"
        print(f"  Final: center=({xc:.1f}, {yc:.1f}), radius={r:.1f}")
    elif ellipse_score > circle_score and best_ellipse_coverage > 0.9:
        print(f"  → Chose ELLIPSE (better score: {ellipse_score:.3f} > {circle_score:.3f})")
        (cx, cy), (MA, ma), angle = best_ellipse
        cv2.ellipse(final_mask, best_ellipse, 255, -1)
        shape_type = "ellipse"
        print(f"  Final: center=({cx:.1f}, {cy:.1f}), axes=({MA/2:.1f}, {ma/2:.1f}), angle={angle:.1f}°")
    else:
        print(f"  → Chose CIRCLE (better score: {circle_score:.3f} >= {ellipse_score:.3f})")
        xc, yc, r = best_circle
        Y, X = np.ogrid[:H, :W]
        final_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        shape_type = "circle"
        print(f"  Final: center=({xc:.1f}, {yc:.1f}), radius={r:.1f}")
    
    return final_mask, edges, shape_type

# EVALUATION

def calculate_metrics(predicted_mask, ground_truth_mask):
    """Calculate TPR, FPR, Accuracy, IoU"""


    pred = (predicted_mask.flatten() > 127).astype(int)
    gt = (ground_truth_mask.flatten() > 127).astype(int)
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return tpr, fpr, accuracy, iou


# TESTING

def test_on_single_image(image_path, ground_truth_path=None, save_prefix='result'):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None

    print(f"\n{'='*70}")
    print(f"Testing on: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"{'='*70}")



    print("Applying Method 3: Combined")
    mask3, _,_ = color_then_circle_or_ellipse(image)

    # Load ground truth
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = cv2.imread(ground_truth_path, 0)


        # Method 3 metrics
        tpr3, fpr3, acc3, iou3 = calculate_metrics(mask3, ground_truth)
        print(f"\nMethod 3: Combined")
        print(f"  TPR: {tpr3:.4f} | FPR: {fpr3:.4f} | Accuracy: {acc3:.4f} | IoU: {iou3:.4f}")

    # # Visualization
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # # ORIGINAL IMAGE
    # axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    # axes[0, 0].axis('off')

    # # METHOD 3
    # axes[0, 1].imshow(mask3, cmap='gray')
    # axes[0, 1].set_title('Method 3: Combined', fontsize=12, fontweight='bold')
    # axes[0, 1].axis('off')




    # overlay = image.copy()
    # overlay[mask3 > 127] = [0, 255, 0]
    # blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    # axes[1, 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    # axes[1, 1].set_title('Overlay', fontsize=12, fontweight='bold')
    # axes[1, 1].axis('off')

    # # Ground Truth panel
    # if ground_truth is not None:
    #     axes[1, 0].imshow(ground_truth, cmap='gray')
    #     axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    #     axes[1, 0].axis('off')
    # else:
    #     axes[1, 0].axis('off')

    # plt.tight_layout()


    # Visualization
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))

    overlay = image.copy()
    overlay[mask3 > 127] = [0, 255, 0]
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    axes.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes.set_title('Overlay', fontsize=12, fontweight='bold')
    axes.axis('off')


    plt.tight_layout()
    save_path = f'results/task2/{save_prefix}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

    return mask3
    


# MAIN
if __name__ == "__main__":
    
    
    img_dir = "final_dataset_small/Easy/images"
    mask_dir = "final_dataset_small/Easy/masks"

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    for i in range(len(image_files)):

            mask3 = test_on_single_image(image_files[i], mask_files[i], save_prefix=image_files[i][-10:])

        # Generate individual ROC curves for each method
    
    print(" TEST COMPLETED!")