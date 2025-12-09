import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel
from skimage.measure import ransac
import glob

#circle and ellipse--all test cases
# restriction: 1. not for small objects   2. rely on picture quality
#designed for task1


# use circle model for circles, use contour for ellipse

# METHOD 3: color, then choose between circle/ellipse

def color_then_circle(image):

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
        return ocean_mask
    
    points = np.column_stack([xs, ys])

    # RANSAC circle fitting on edge points
    model = CircleModel()
    try:
        circle, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=2, max_trials=2000)
        xc, yc, r = circle.params
        circle_center=[xc,yc]
    except Exception as e:
        print(f"Circle fit failed: {e}, fallback to mask")
        return ocean_mask
        
    try:
        # Calculate circle fitting error
        distances_circle = np.abs(np.sqrt((xs - xc)**2 + (ys - yc)**2) - r)
        circle_error = np.mean(distances_circle)
        
    except np.linalg.LinAlgError:
        circle_error = float('inf')
    


    # FIT ELLIPSE 
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found")
        return np.zeros_like(gray)
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
        # Need at least 5 points to fit ellipse
    if len(contour) < 5:
        print("Insufficient points for shape fitting")
        return np.zeros_like(gray)
    
    # Squeeze contour to get (N, 2) shape
    pts = contour.squeeze()
    if pts.ndim != 2:
        print("Invalid contour shape")
        return np.zeros_like(gray)
    
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)

    try:
        ellipse = cv2.fitEllipse(contour)

        # Calculate ellipse fitting error (Euclidean distance in pixels)
        (cx, cy), (MA, ma), angle = ellipse # center, major/minor axis length, rotation angle
        a = MA / 2
        b = ma / 2
        angle_rad = np.deg2rad(angle)
        ellipse_center=[cx,cy]

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
    print(f"  Circle error: {circle_error:.2f}, Ellipse error: {ellipse_error:.2f}")
    print(f"  Circle Center: {circle_center}, Ellipse center: {ellipse_center}")

    H, W = ocean_mask.shape
    final_mask = np.zeros_like(ocean_mask)
    
    if circle_error > ellipse_error*2 and ellipse_error > 9.5:
        print("  → Chose ELLIPSE fit")
        cv2.ellipse(final_mask, ellipse, 255, -1)
        center = ellipse_center
    else:
        print("  → Chose CIRCLE fit")
        Y, X = np.ogrid[:H, :W]
        final_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255
        center = circle_center
    
    return final_mask, center


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
    mask3, center = color_then_circle(image)

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

    return mask3,center
    


# MAIN
if __name__ == "__main__":
    
    
    img_dir = "final_dataset_small/Easy/images"
    mask_dir = "final_dataset_small/Easy/masks"

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    for i in range(len(image_files)):

            mask3, center = test_on_single_image(image_files[i], mask_files[i], save_prefix=image_files[i][-10:])
            print(center)
        # Generate individual ROC curves for each method
    
    print(" TEST COMPLETED!")