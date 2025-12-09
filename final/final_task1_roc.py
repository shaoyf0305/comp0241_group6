import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel, ransac
from sklearn.metrics import roc_curve, auc
import glob
from scipy.optimize import curve_fit

# task1 roc framework
# what parameters to tune need to be concerned further
# method3 radius scale make no sense, will be replaced
# should do it after the method is fully decided

def roc_model(x, a):
    return 1 - np.exp(-a * x)


# METHOD 1: Vary canny edge_value, lower threshold = stricter edge
def segment_by_geometry(img, edge_value=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)

    edges = cv2.Canny(gray, edge_value-100, edge_value)
    ys, xs = np.where(edges > 0)
    points = np.column_stack([xs, ys])

    model = CircleModel()
    try:
        circle, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=2, max_trials=2000)
        xc, yc, r = circle.params
    except:
        return np.zeros_like(gray)

    H, W = gray.shape
    Y, X = np.ogrid[:H, :W]
    mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255

    return mask


# METHOD 2: Vary HSV Hue upper bound, higher = more permissive
def segment_by_color(img, hue_upper=15):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([135, 240+hue_upper, 240+hue_upper])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    sure_fg = cv2.erode(clean, kernel, iterations=3)
    sure_bg = cv2.dilate(clean, kernel, iterations=6)
    sure_bg = cv2.threshold(sure_bg, 1, 255, cv2.THRESH_BINARY_INV)[1]

    markers = np.zeros_like(sure_fg, dtype=np.int32)
    markers[sure_bg == 255] = 1
    markers[sure_fg == 255] = 2

    img_copy = img.copy()
    cv2.watershed(img_copy, markers)

    seg = (markers == 2).astype(np.uint8) * 255

    return seg

# METHOD 3: color, then choose between circle/ellipse

def color_then_circle(image , radius_scale=1.05):
    # Color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([135, 255, 255])
    ocean_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Smooth mask, remove land + noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply Canny edge detection on the color mask
    edges = cv2.Canny(ocean_mask, 50, 150)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        print("No contours found, fallback to color mask only")
        return ocean_mask
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Need at least 5 points to fit ellipse
    if len(contour) < 5:
        print("Insufficient points for shape fitting, fallback to mask")
        return ocean_mask
    
    # Squeeze contour to get (N, 2) shape
    pts = contour.squeeze()
    if pts.ndim != 2:
        print("Invalid contour shape, fallback to mask")
        return ocean_mask
    
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    
    # FIT CIRCLE 
    try:
        circle_model, inliers = ransac( pts, CircleModel, min_samples=3, residual_threshold=2,max_trials=1000)
        xc, yc, r = circle_model.params
        
        # Calculate circle fitting error (Euclidean distance in pixels)
        distances_circle = np.abs(np.sqrt((x - xc)**2 + (y - yc)**2) - r)
        circle_error = np.mean(distances_circle)
        
    except Exception as e:
        print(f"Circle fitting failed: {e}")
        circle_error = float('inf')
    
    # FIT ELLIPSE 
    try:
        ellipse = cv2.fitEllipse(contour)

        # Calculate ellipse fitting error (Euclidean distance in pixels)
        (cx, cy), (MA, ma), angle = ellipse # center, major/minor axis length, rotation angle
        a = MA / 2
        b = ma / 2
        angle_rad = np.deg2rad(angle)
        
        distances_ellipse = []
        for xi, yi in zip(x, y):
            xp = xi - cx
            yp = yi - cy
            
            xr = xp * np.cos(angle_rad) + yp * np.sin(angle_rad)
            yr = -xp * np.sin(angle_rad) + yp * np.cos(angle_rad)
            
            dist_from_center = np.sqrt(xr**2 + yr**2)
            theta = np.arctan2(yr, xr)
            ellipse_radius = (a * b) / np.sqrt((b * np.cos(theta))**2 + (a * np.sin(theta))**2)
            dist_to_ellipse = np.abs(dist_from_center - ellipse_radius)
            distances_ellipse.append(dist_to_ellipse)
        
        ellipse_error = np.mean(distances_ellipse)
        
    except Exception as e:
        print(f"Ellipse fitting failed: {e}, fallback to mask")
        return ocean_mask
    
    # Choose the better fit
    print(f"  Circle error: {circle_error:.2f} px, Ellipse error: {ellipse_error:.2f} px")
    
    H, W = ocean_mask.shape
    final_mask = np.zeros_like(ocean_mask)
    
    if circle_error < ellipse_error*3.5:
        print("  → Chose CIRCLE fit")

        # apply scale
        r = r * radius_scale

        Y, X = np.ogrid[:H, :W]
        final_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
    
    else:
        print("  → Chose ELLIPSE fit")

        # unpack ellipse
        (cx, cy), (MA, ma), angle = ellipse

        # apply scale
        MA *= radius_scale
        ma *= radius_scale

        # draw scaled ellipse
        cv2.ellipse(final_mask, ((cx, cy), (MA, ma), angle), 255, -1)
    
    return final_mask


def compute_roc_for_method(method_func, param_name, param_values, test_images_dir, test_masks_dir):

    image_files = sorted(glob.glob(os.path.join(test_images_dir, '*.*')))
    
    tpr_list = []
    fpr_list = []
    param_list = []
    
    print(f"Processing {len(image_files)} images for each parameter value...")
    
    for param_val in param_values:
        all_predictions = []
        all_ground_truth = []
        
        print(f"  Testing {param_name} = {param_val}", end='')
        
        success_count = 0
        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            img_name = os.path.basename(img_path)
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(test_masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            ground_truth = cv2.imread(mask_path, 0)
            if ground_truth is None:
                continue
            
            try:
                # Apply method with specific parameter value
                if param_name == 'edge_value':
                    mask = method_func(image, edge_value=param_val)
                elif param_name == 'hue_upper':
                    mask = method_func(image, hue_upper=param_val)
                elif param_name == 'radius_scale':
                    mask = method_func(image, radius_scale=param_val)
                
                # Convert to binary
                pred = (mask.flatten() > 127).astype(int)
                gt = (ground_truth.flatten() > 127).astype(int)
                
                all_predictions.append(pred)
                all_ground_truth.append(gt)
                success_count += 1
                
            except Exception as e:
                continue
        
        print(f" ({success_count} images processed)")
        
        if len(all_predictions) == 0:
            continue
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions)
        all_ground_truth = np.concatenate(all_ground_truth)
        
        # Calculate TPR and FPR for this parameter value
        tp = np.sum((all_predictions == 1) & (all_ground_truth == 1))
        fp = np.sum((all_predictions == 1) & (all_ground_truth == 0))
        tn = np.sum((all_predictions == 0) & (all_ground_truth == 0))
        fn = np.sum((all_predictions == 0) & (all_ground_truth == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0


        tpr_list.append(tpr)
        fpr_list.append(fpr)
        param_list.append(param_val)
        
        print(f"    → TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    
    # Sort by FPR for proper ROC curve
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    tpr_sorted = np.array(tpr_list)[sorted_indices]
    param_sorted = np.array(param_list)[sorted_indices]
    

    # Calculate AUC, area under ROC
    auc_score = auc(fpr_sorted, tpr_sorted) if len(fpr_sorted) > 1 else 0
    
    return fpr_sorted, tpr_sorted, param_sorted, auc_score 


def plot_individual_roc_curves(test_images_dir, test_masks_dir):
    
    print("PARAMETRIC ROC CURVE ANALYSIS")
    
    # Define parameter ranges
    # METHOD 1: RANSAC residual threshold
    edge_values = [ 100, 120 , 140 ,160 ,180, 200]
    
    # METHOD 2: blue extent
    hue_uppers = [-15, -10, -5, 0, 5 , 10, 15]
     
    # METHOD 3: Radius scaling
    radius_scales = [0.85, 0.90, 0.93, 0.95, 0.98, 1.0, 1.02, 1.05, 1.08, 1.10, 1.15]
    

    # METHOD 1 
    print("METHOD 1: GEOMETRY-BASED (Canny + RANSAC Circle Fitting)")
    fpr1, tpr1, params1, auc1 = compute_roc_for_method(segment_by_geometry, 'edge_value', edge_values, test_images_dir, test_masks_dir)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(fpr1, tpr1, 'bo-', linewidth=2.5, markersize=10, label=f'Raw points (AUC={auc1:.4f})')

    # Annotate every raw point
    for x, y, p in zip(fpr1, tpr1, params1): 
        ax1.annotate(f"{p}", (x, y), textcoords="offset points", xytext=(5, 5),fontsize=8)

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax1.set_title('ROC Curve: Method 1 (Geometry)\nVarying Canny edge threshold', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_method1_geometry_fitted.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Method 1 ROC curve saved to 'roc_method1_geometry.png'")
    plt.show()


    
    # METHOD 2 
    print("METHOD 2: COLOR-BASED (HSV Thresholding + Watershed)")
    fpr2, tpr2, params2, auc2 = compute_roc_for_method(segment_by_color, 'hue_upper', hue_uppers, test_images_dir, test_masks_dir)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(fpr2, tpr2, 'rs-', linewidth=2.5, markersize=10, label=f'AUC = {auc2:.4f}')
    
    # # Annotate some key points
    # for i in [0, -1]: #[0, len(params2)//2, -1]:
    #     if i < len(params2):
    #         ax2.annotate(f'H={int(params2[i])}', xy=(fpr2[i], tpr2[i]),  xytext=(10, -10), textcoords='offset points', fontsize=9, 
    #                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Annotate every raw point
    for x, y, p in zip(fpr2, tpr2, params2): 
        ax2.annotate(f"{p}", (x, y), textcoords="offset points", xytext=(5, 5),fontsize=8)
    
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax2.set_title('ROC Curve: Method 2 (Color)\nVarying HSV hue_upper bound', fontsize=16, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_method2_color.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Method 2 ROC curve saved to 'roc_method2_color.png'")
    plt.show()
    
    # METHOD 3 
    print("METHOD 3: COMBINED (Color + Canny + RANSAC Circle)")
    fpr3, tpr3, params3, auc3 = compute_roc_for_method( color_then_circle, 'radius_scale', radius_scales, test_images_dir, test_masks_dir)

    # Plot Method 3
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(fpr3, tpr3, 'g^-', linewidth=2.5, markersize=10, label=f'AUC = {auc3:.4f}')
    
    # Annotate every raw point
    for x, y, p in zip(fpr3, tpr3, params3): 
        ax3.annotate(f"{p}", (x, y), textcoords="offset points", xytext=(5, 5),fontsize=8)
            
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.02])
    ax3.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax3.set_title('ROC Curve: Method 3 (Combined)\nVarying radius_scale factor', fontsize=16, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=12)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_method3_combined.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Method 3 ROC curve saved to 'roc_method3_combined.png'")
    plt.show()
    
    # Print summary
    print(f"Method 1 (Geometry):  AUC = {auc1:.4f}")
    print(f"Method 2 (Color):     AUC = {auc2:.4f}")
    print(f"Method 3 (Combined):  AUC = {auc3:.4f}")

# MAIN
if __name__ == "__main__":
    
    # Set your test dataset paths here
    test_images_dir = "final_dataset_small/Easy/images"
    test_masks_dir = "final_dataset_small/Easy/masks"
    
    # test_images_dir = "final_dataset/Test/images"
    # test_masks_dir = "final_dataset/Test/masks"
    
    # Check if directories exist
    if not os.path.exists(test_images_dir):
        print(f"Error: Directory not found: {test_images_dir}")
        print("Please update the paths in the script!")
    elif not os.path.exists(test_masks_dir):
        print(f"Error: Directory not found: {test_masks_dir}")
        print("Please update the paths in the script!")
    else:
        # Generate individual ROC curves for each method
        plot_individual_roc_curves(test_images_dir, test_masks_dir)
    
    print("\n analysis completed!")
