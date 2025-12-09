import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel
from skimage.measure import ransac
import glob

#circle--all test cases
# restriction: 1. not for small objects   2. not for distorted graph
# apply:case_all_circle




def color_then_circle(image, center_weight=0.2, auto_scale_margin=20, use_connected_components=True):
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
    center=[xc,yc]
    return circle_mask, center, edges


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
    mask3, center,_ = color_then_circle(image)

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

    axes.imshow(cv2.cvtColor(blended, cv2._BGR2RGB))
    axes.set_title('Overlay', fontsize=12, fontweight='bold')
    axes.axis('off')


    plt.tight_layout()
    save_path = f'results/task2/{save_prefix}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

    return mask3,center
    


# MAIN
if __name__ == "__main__":
    
    
    img_dir = "final_dataset/Easy/images"
    mask_dir = "final_dataset/Easy/masks"

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    for i in range(len(image_files)):

            mask3, center = test_on_single_image(image_files[i], mask_files[i], save_prefix=image_files[i][-10:])
            print(center)
        # Generate individual ROC curves for each method
    
    print(" TEST COMPLETED!")