import pandas as pd
import numpy as np
import cv2

def buffer_and_calculate_overlap(img1, img2, buffer_value):
    """
    Buffers two binary masks and calculates the overlapping area.

    Args:
        img1 (PIL.Image.Image): First binary mask.
        img2 (PIL.Image.Image): Second binary mask.
        buffer_value (int): Buffer size in pixels for dilation.

    Returns:
        int: Overlapping area in pixels.
    """
    # Convert PIL images to numpy arrays
    mask1 = np.array(img1)
    mask2 = np.array(img2)
    
    # Ensure binary masks are valid (convert if necessary)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    # Create structuring element for buffering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_value, buffer_value))
    
    # Buffer the masks using dilation
    buffered_mask1 = cv2.dilate(mask1, kernel, iterations=1)
    buffered_mask2 = cv2.dilate(mask2, kernel, iterations=1)
    
    # Calculate the overlapping area
    overlap = np.logical_and(buffered_mask1, buffered_mask2)
    overlapping_area = np.sum(overlap)
    print(overlapping_area)
    
    return overlapping_area