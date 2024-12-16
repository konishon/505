import numpy as np
from PIL import Image
import cv2

def combine_masks_and_clip_image(image_path, mask_array, output_path):
    """
    Combines multiple masks, resizes them to match the image, uses the combined mask to clip the image, and saves the result.

    Args:
        image_path (str): Path to the original image.
        mask_array (list of np.ndarray): List of binary masks as numpy arrays.
        output_path (str): Path to save the clipped image.
    """
    # Load the original image
    original_image = np.array(Image.open(image_path))
    image_height, image_width = original_image.shape[:2]

    # Initialize a combined mask
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Resize each mask to match the dimensions of the original image and combine
    for mask in mask_array:
        resized_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.logical_or(combined_mask, resized_mask).astype(np.uint8)

    # Apply the combined mask to the original image
    if len(original_image.shape) == 3:  # For RGB images
        combined_mask_3d = np.stack([combined_mask] * 3, axis=-1)
        clipped_image = np.where(combined_mask_3d, original_image, 0)
    else:  # For grayscale images
        clipped_image = np.where(combined_mask, original_image, 0)

    # Save the resulting image
    clipped_image_pil = Image.fromarray(clipped_image)
    clipped_image_pil.save(output_path)
    print(f"Clipped image saved to: {output_path}")

# Example usage
image_path = "data/row_0_col_1638/original_image.png"
mask1 = np.array(Image.open("data/row_0_col_1638/region_9_with_border.png").convert("L")) > 0
mask2 = np.array(Image.open("data/row_0_col_1638/region_12_with_border.png").convert("L")) > 0

# Create a list of masks
masks = [mask1.astype(np.uint8), mask2.astype(np.uint8)]
output_path = "clipped_image.png"

combine_masks_and_clip_image(image_path, masks, output_path)
