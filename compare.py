import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import load_region_images
import pandas as pd
from tqdm import tqdm

def process_masks_and_calculate_overlap(mask_path1, mask_path2, buffer_size=3):
    """
    Loads two masks from file paths, buffers them (expands white regions), 
    and calculates their overlapping area.

    Args:
        mask_path1 (str): Path to the first binary mask image.
        mask_path2 (str): Path to the second binary mask image.
        buffer_size (int): Size of the buffer (dilation kernel). Default is 3.

    Returns:
        dict: A dictionary containing:
            - 'overlapping_area': Number of overlapping pixels.
            - 'buffered_mask1': Buffered mask 1.
            - 'buffered_mask2': Buffered mask 2.
            - 'overlap_mask': Overlap between the buffered masks.
    """
    # Load masks as grayscale
    mask1 = np.array(Image.open(mask_path1).convert("L"))
    mask2 = np.array(Image.open(mask_path2).convert("L"))

    # Log basic mask info
    #print(f"Loaded Mask 1: {mask1.shape}, Unique values: {np.unique(mask1)}")
    #print(f"Loaded Mask 2: {mask2.shape}, Unique values: {np.unique(mask2)}")

    # Threshold masks to binary (0 or 1)
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    # Check thresholded unique values
    #print("Thresholded Mask 1 unique values:", np.unique(mask1_binary))
    #print("Thresholded Mask 2 unique values:", np.unique(mask2_binary))

    # Buffer masks using dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_size, buffer_size))
    buffered_mask1 = cv2.dilate(mask1_binary, kernel, iterations=1)
    buffered_mask2 = cv2.dilate(mask2_binary, kernel, iterations=1)

    # Check buffered mask values
    #print("Buffered Mask 1 unique values:", np.unique(buffered_mask1))
    #print("Buffered Mask 2 unique values:", np.unique(buffered_mask2))

    # Calculate overlap
    overlap_mask = np.logical_and(buffered_mask1, buffered_mask2).astype(np.uint8)
    overlapping_area = np.sum(overlap_mask)

    # Log the results
    #print(f"Overlapping Area: {overlapping_area} pixels")

    # Return results as a dictionary
    return {
        "overlapping_area": overlapping_area,
        "buffered_mask1": buffered_mask1,
        "buffered_mask2": buffered_mask2,
        "overlap_mask": overlap_mask
    }


def visualize_results(result):
    """
    Visualizes the buffered masks, overlap, and combined masks.

    Args:
        result (dict): A dictionary containing buffered masks and overlap mask.
    """
    # Visualization using Matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(result["buffered_mask1"], cmap="gray")
    axes[0].set_title("Buffered Mask 1")
    axes[0].axis("off")

    axes[1].imshow(result["buffered_mask2"], cmap="gray")
    axes[1].set_title("Buffered Mask 2")
    axes[1].axis("off")

    axes[2].imshow(result["overlap_mask"], cmap="gray")
    axes[2].set_title(f"Overlap Area: {result['overlapping_area']} pixels")
    axes[2].axis("off")

    combined_mask = result["buffered_mask1"] + result["buffered_mask2"]
    axes[3].imshow(combined_mask, cmap="gray")
    axes[3].set_title("Combined Buffered Masks")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

def main2():    
    image_path = "data/row_0_col_1638/original_image.png"
    folder_path = "data/row_0_col_1638"
    region_images_path = load_region_images(folder_path)
    
    similarity_matrix = pd.DataFrame(0, index=region_images_path, columns=region_images_path)
    for i, mask_path1 in tqdm(enumerate(region_images_path), desc="Processing Regions"):
        for j, mask_path2 in enumerate(region_images_path[i:], start=i):
            if mask_path1 == mask_path2:
                continue  # Skip self-comparison
            buffer_size = 20

            result = process_masks_and_calculate_overlap(mask_path1, mask_path2, buffer_size)
            overlapping_area = result['overlapping_area']
            similarity_matrix.loc[region_images_path[i], region_images_path[j]] = overlapping_area
            similarity_matrix.loc[region_images_path[j], region_images_path[i]] = overlapping_area
    #print(similarity_matrix)
    similarity_matrix.to_csv("sim.csv")
    
# if __name__ == "__main__":
#     main2()


if __name__ == "__main__":
    mask_path1 = "data/row_0_col_1638/region_3_with_border.png"  # Replace with actual path
    mask_path2 = "data/row_0_col_1638/region_1_with_border.png"  # Replace with actual path

    buffer_size = 10  # Adjust based on the level of exconpansion needed

    result = process_masks_and_calculate_overlap(mask_path1, mask_path2, buffer_size)

    visualize_results(result)
