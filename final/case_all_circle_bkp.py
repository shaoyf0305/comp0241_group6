import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel
from skimage.measure import ransac

#circle--all self taken photos
#designed for task2--optimal

# METHOD 1

def segment_by_geometry(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)

    # Edges
    edges = cv2.Canny(gray, 80, 180)
    ys, xs = np.where(edges > 0)
    points = np.column_stack([xs, ys])

    # RANSAC circle
    model = CircleModel()
    circle, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=2, max_trials=2000)

    xc, yc, r = circle.params

    # Build final mask
    H, W = gray.shape
    Y, X = np.ogrid[:H, :W]
    mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255

    return mask

# METHOD 2

def segment_by_color(img):

    # HSV threshold 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Hue/Saturation/Value
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Smooth mask 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Markers for watershed 
    sure_fg = cv2.erode(clean, kernel, iterations=3) # Foreground = globe
    sure_bg = cv2.dilate(clean, kernel, iterations=6) # Background = not-globe
    sure_bg = cv2.threshold(sure_bg, 1, 255, cv2.THRESH_BINARY_INV)[1]

    # Label markers
    markers = np.zeros_like(sure_fg, dtype=np.int32)
    markers[sure_bg == 255] = 1
    markers[sure_fg == 255] = 2

    # Watershed 
    img_copy = img.copy()
    cv2.watershed(img_copy, markers)

    # Region with marker==2 corresponds to globe (foreground)
    seg = (markers == 2).astype(np.uint8) * 255

    # Optional round boundary
    # seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel, iterations=2)

    return seg


# METHOD 3: Connected Component Based Circle Fitting

def color_then_circle(image, center_weight=0.3, auto_scale_margin=20, use_contour_filter=True, 
                     use_edge_guided_fill=True, use_connected_components=True):
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
    use_contour_filter : bool
        If True, filter edges to only keep the outer boundary contour
    use_edge_guided_fill : bool
        If True, use fitted circle to guide filling of the color mask
    use_connected_components : bool
        If True, use connected components analysis to keep all central data points
    
    Returns:
    --------
    circle_mask : numpy array     Binary mask of the detected circle
    edges : numpy array      Edge detection result (for visualization)
    """

    H, W = image.shape[:2]
    img_center_x = W / 2
    img_center_y = H / 2
    max_distance = np.sqrt(img_center_x**2 + img_center_y**2)  # diagonal distance
    max_possible_radius = min(H, W) / 2  # Maximum reasonable radius

    # Color threshold, ocean only
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
        center_roi = np.zeros_like(ocean_mask)
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
        
        print(f"  Found {len(central_components)} central components out of {num_labels-1} total")
        
        if len(central_components) == 0:
            print("  No central components found, trying less strict filter...")
            # Fallback: try less strict
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                area = stats[i, cv2.CC_STAT_AREA]
                dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                if area > 100 and dist < max_distance * 0.6:
                    central_components.append(i)
            print(f"  Found {len(central_components)} with relaxed filter")
        
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
            
    elif use_contour_filter:
        # Find contours and keep only the largest outer boundary
        contours, _ = cv2.findContours(ocean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            print("No contours found, using direct edge detection")
            edges = cv2.Canny(ocean_mask, 50, 150)
        else:
            # Find the largest contour (should be the globe boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create edge image from only the outer contour
            edges = np.zeros_like(ocean_mask)
            cv2.drawContours(edges, [largest_contour], -1, 255, 1)
            
            print(f"  Filtered to largest contour with {len(largest_contour)} points")
        
        points_ys, points_xs = np.where(edges > 0)
    else:
        # Apply Canny edge detection on the color mask (original method)
        edges = cv2.Canny(ocean_mask, 50, 150)
        points_ys, points_xs = np.where(edges > 0)
    
    # Extract edge points for circle fitting
    if len(points_xs) < 20:
        print("Insufficient edge points, fallback to color mask only")
        return ocean_mask, edges
    
    points = np.column_stack([points_xs, points_ys])
    print(f"  Using {len(points)} points for circle fitting")

    # RANSAC circle fitting with multiple trials to find LARGEST circle that contains the data
    model = CircleModel()
    best_circle = None
    best_score = -np.inf
    n_iterations = 10  # More iterations for better results
    
    for i in range(n_iterations):
        try:
            circle, inliers = ransac(points, CircleModel, min_samples=3, 
                                    residual_threshold=3, max_trials=2000)
            xc, yc, r = circle.params
            
            # For connected components mode: we want the LARGEST circle that contains most points
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
            combined_score = (0.4 * coverage_ratio + 
                            0.4 * radius_score + 
                            0.2 * center_score)
            
            print(f"  Trial {i+1}: center=({xc:.1f}, {yc:.1f}), r={r:.1f}, "
                    f"coverage={coverage_ratio:.3f}, radius_score={radius_score:.3f}, "
                    f"score={combined_score:.3f}")

            
            if combined_score > best_score:
                best_score = combined_score
                best_circle = (xc, yc, r)
                
        except Exception as e:
            print(f"  Trial {i+1} failed: {e}")
            continue
    
    if best_circle is None:
        print("All circle fits failed, fallback to mask")
        return ocean_mask, edges
    
    xc, yc, r = best_circle
    distance_from_center = np.sqrt((xc - img_center_x)**2 + (yc - img_center_y)**2)
    
    print(f"\n  BEST CIRCLE SELECTED:")
    print(f"    Center: ({xc:.1f}, {yc:.1f})")
    print(f"    Radius: {r:.1f} (normalized: {r/max_possible_radius:.3f})")
    print(f"    Image center: ({img_center_x:.1f}, {img_center_y:.1f})")
    print(f"    Distance from image center: {distance_from_center:.1f} pixels")
    print(f"    Final score: {best_score:.3f}")

    if use_edge_guided_fill:
        # Strategy: Use the fitted circle to guide filling, but respect color information
        # 1. Create a perfect circle mask from the fitted parameters
        r=r+auto_scale_margin
        Y, X = np.ogrid[:H, :W]
        fitted_circle_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        
        # 2. Combine with color mask: keep areas that are EITHER in color mask OR in fitted circle
        # This fills in gaps where continents broke the color segmentation
        combined_mask = cv2.bitwise_or(ocean_mask, fitted_circle_mask)
        
        # 3. But constrain to only within the fitted circle boundary
        # This prevents extending beyond the detected circle
        final_mask = cv2.bitwise_and(combined_mask, fitted_circle_mask)
        
        # 4. Smooth the result to create a nice circular boundary
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=3)
        
        # 5. Final circle constraint to ensure perfect circular boundary
        final_mask = cv2.bitwise_and(final_mask, fitted_circle_mask)
        
        print(f"  Edge-guided fill: Used circle boundary to complete the mask")
        
        return final_mask, edges
    else:
        # Build final circle mask (original method - pure geometric circle)
        Y, X = np.ogrid[:H, :W]
        circle_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        return circle_mask, edges


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

def test_on_single_image(image_path, ground_truth_path=None, save_prefix='result', 
                         center_weight=0.3, auto_scale_margin=5, use_contour_filter=True,
                         use_edge_guided_fill=True, use_connected_components=True):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None

    print(f"\n{'='*70}")
    print(f"Testing on: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Center weight: {center_weight}")
    print(f"Auto-scale margin: {auto_scale_margin} pixels")
    print(f"Use contour filter: {use_contour_filter}")
    print(f"Use edge-guided fill: {use_edge_guided_fill}")
    print(f"Use connected components: {use_connected_components}")
    print(f"{'='*70}")

    print("Applying Method 1: Geometry")
    mask1 = segment_by_geometry(image)

    print("Applying Method 2: Color")
    mask2 = segment_by_color(image)

    print("Applying Method 3: Combined (with center penalty)")
    mask3, edges3 = color_then_circle(image, center_weight=center_weight, 
                                      auto_scale_margin=auto_scale_margin,
                                      use_contour_filter=use_contour_filter,
                                      use_edge_guided_fill=use_edge_guided_fill,
                                      use_connected_components=use_connected_components)



    # Load ground truth
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = cv2.imread(ground_truth_path, 0)

        # Method 1 metrics
        tpr1, fpr1, acc1, iou1 = calculate_metrics(mask1, ground_truth)
        print(f"\nMethod 1: Geometry")
        print(f"  TPR: {tpr1:.4f} | FPR: {fpr1:.4f} | Accuracy: {acc1:.4f} | IoU: {iou1:.4f}")

        # Method 2 metrics
        tpr2, fpr2, acc2, iou2 = calculate_metrics(mask2, ground_truth)
        print(f"\nMethod 2: Color")
        print(f"  TPR: {tpr2:.4f} | FPR: {fpr2:.4f} | Accuracy: {acc2:.4f} | IoU: {iou2:.4f}")

        # Method 3 metrics
        tpr3, fpr3, acc3, iou3 = calculate_metrics(mask3, ground_truth)
        print(f"\nMethod 3: Combined")
        print(f"  TPR: {tpr3:.4f} | FPR: {fpr3:.4f} | Accuracy: {acc3:.4f} | IoU: {iou3:.4f}")


    # Visualization

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    # ROW 1: Original and masks
    # ORIGINAL IMAGE
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # METHOD 1
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title('Method 1: Geometry', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # METHOD 2
    axes[0, 2].imshow(mask2, cmap='gray')
    axes[0, 2].set_title('Method 2: Color', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # METHOD 3
    axes[0, 3].imshow(mask3, cmap='gray')
    axes[0, 3].set_title(f'Method 3: Combined (auto-scaled)', 
                         fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    # ROW 2: Edge detection visualization
    # Ground Truth panel
    if ground_truth is not None:
        axes[1, 0].imshow(ground_truth, cmap='gray')
        axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')
    
    # Show edge points detected in Method 3
    # Create visualization with edge points on original image
    edge_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    edge_points_y, edge_points_x = np.where(edges3 > 0)
    edge_viz[edge_points_y, edge_points_x] = [255, 0, 0]  # Red dots for edge points
    
    axes[1, 1].imshow(edge_viz)
    axes[1, 1].set_title(f'Method 3: Edge Points ({len(edge_points_x)} points)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Show just the edges in white on black
    axes[1, 2].imshow(edges3, cmap='gray')
    axes[1, 2].set_title('Method 3: Canny Edges', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Empty slot
    axes[1, 3].axis('off')

    # ROW 3: Overlays for visualization
    method_list = [(mask1, 'Geometry'), (mask2, 'Color'), (mask3, 'Combined')]

    for idx, (mask, name) in enumerate(method_list):
        overlay = image.copy()
        overlay[mask > 127] = [0, 255, 0]    # paint segmented region green
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        axes[2, idx + 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2, idx + 1].set_title(f'{name} Overlay', fontsize=12, fontweight='bold')
        axes[2, idx + 1].axis('off')
    
    axes[2, 0].axis('off')

    plt.tight_layout()

    save_path = f'{save_prefix}_extraction_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

    return mask1, mask2, mask3



# MAIN

if __name__ == "__main__":

    print(" TESTING ON 000000.png")
    test_image = "final_dataset/Test/images/000003.jpg"
    # test_image = "final_dataset/Test/images/000000.jpg"

    test_mask = "final_dataset/Test/masks/000003.jpg"
    # test_mask = "final_dataset/Easy/masks/000016.png"

    if os.path.exists(test_image):
        # You can adjust parameters:
        # center_weight: 0=no penalty, 0.5=balanced, 1.0=maximum center preference
        # auto_scale_margin: pixel margin beyond farthest point (default: 5 pixels)
        # use_contour_filter: True=only use outer boundary, False=all edges
        # use_edge_guided_fill: True=fill broken areas using circle, False=pure geometric circle
        # use_connected_components: True=find central components and fit largest circle
        mask1, mask2, mask3 = test_on_single_image(test_image, test_mask, 
                                                    save_prefix='easy_00',
                                                    center_weight=0.2,
                                                    auto_scale_margin=20,
                                                    use_contour_filter=False,
                                                    use_edge_guided_fill=True,
                                                    use_connected_components=True)
    else:
        print(f"Image not found: {test_image}")
        print("Please update the path!")
    
    print(" TEST COMPLETED!")