import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_compactness_ratio(binary_masks):
    """
    Calculate the compactness ratio (Perimeter/Area) of combined binary masks.

    Args:
        binary_masks (list of np.ndarray): List of binary masks (2D NumPy arrays).

    Returns:
        float: Compactness ratio (Perimeter/Area).
        np.ndarray: Combined binary mask.
    """
    # Combine all binary masks using logical OR
    combined_mask = np.logical_or.reduce(binary_masks).astype(np.uint8) * 255  # Convert to 0-255 scale

    # Find external contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If no contours found, return infinity
        return float('inf'), combined_mask

    # Assume the largest contour is the shape we are analyzing
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter of the largest contour
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    
    # Avoid division by zero
    if area == 0:
        return float('inf'), combined_mask
    
    # Calculate compactness ratio (Perimeter / Area)
    compactness_ratio = perimeter / area
    
    return compactness_ratio, combined_mask

def process_masks(mask_paths):
    """
    Process binary masks, compute compactness ratio, and visualize the combined mask.

    Args:
        mask_paths (list of str): List of file paths to the binary mask images.
    """
    # Read masks as grayscale and threshold to binary
    binary_masks = []
    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        binary_masks.append(mask_bin)
    
    # Calculate compactness ratio
    compactness_ratio, combined_mask = calculate_compactness_ratio(binary_masks)
    
    # # Display the combined mask using matplotlib
    # plt.figure(figsize=(8, 8))
    # plt.imshow(combined_mask, cmap='gray')
    # plt.title(f"Combined Mask - Compactness Ratio: {compactness_ratio:.4f}")
    # plt.axis('off')
    # plt.show()
    
    # Print the compactness ratio
    print(f"Compactness Ratio: {compactness_ratio:.4f}")

# File paths to binary mask images
mask_paths = [
    'data/row_0_col_1638/region_9_with_border.png',  # First mask path
    'data/row_0_col_1638/region_12_with_border.png'  # Second mask path
]

# Run the process
if __name__ == "__main__":
    process_masks(mask_paths)


