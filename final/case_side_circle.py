import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import CircleModel
from skimage.measure import ransac

#circle--single test
#designed for task1

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


# METHOD 3: color, then extract circle

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
    except Exception as e:
        print(f"Circle fit failed: {e}, fallback to mask")
        return ocean_mask

    # Build final circle mask
    H, W = ocean_mask.shape
    Y, X = np.ogrid[:H, :W]
    circle_mask = ((X - xc)**2 + (Y - yc)**2 <= r*r).astype(np.uint8) * 255

    return circle_mask


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

    print("Applying Method 1: Geometry")
    mask1 = segment_by_geometry(image)

    print("Applying Method 2: Color")
    mask2 = segment_by_color(image)

    print("Applying Method 3: Combined")
    mask3 = color_then_circle(image)



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

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

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
    axes[0, 3].set_title('Method 3: Combined', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')


    # Overlays for visualization

    method_list = [(mask1, 'Geometry'), (mask2, 'Color'), (mask3, 'Combined')]

    for idx, (mask, name) in enumerate(method_list):
        overlay = image.copy()
        overlay[mask > 127] = [0, 255, 0]    # paint segmented region green
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        axes[1, idx + 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[1, idx + 1].set_title(f'{name} Overlay', fontsize=12, fontweight='bold')
        axes[1, idx + 1].axis('off')

    # Ground Truth panel
    if ground_truth is not None:
        axes[1, 0].imshow(ground_truth, cmap='gray')
        axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')

    plt.tight_layout()

    save_path = f'{save_prefix}_extraction_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()

    return mask1, mask2, mask3



# MAIN

if __name__ == "__main__":

    print(" TESTING ON 000000.png")
    test_image = "final_dataset/Easy/images/000016.png"
    # test_image = "final_dataset/Test/images/000000.jpg"

    test_mask = "final_dataset/Easy/masks/000016.png"
    # test_mask = "final_dataset/Easy/masks/000016.png"

    if os.path.exists(test_image):
        mask1, mask2, mask3 = test_on_single_image(test_image, test_mask, save_prefix='easy_00')
    else:
        print(f"Image not found: {test_image}")
        print("Please update the path!")
    
    print(" TEST COMPLETED!")
