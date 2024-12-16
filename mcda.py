import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from data_loader import load_region_images
from compare import process_masks_and_calculate_overlap


folder_path = "data/row_0_col_1638"
region_images_path = load_region_images(folder_path)

if region_images_path:
    similarity_matrix = pd.DataFrame(0, index=region_images_path, columns=region_images_path)
    for i, mask_path1 in tqdm(enumerate(region_images_path), desc="Processing Regions"):
        for j, mask_path2 in enumerate(region_images_path[i:], start=i):
            if mask_path1 == mask_path2:
                continue  # Skip self-comparison
        
        # Buffer size to expand the white regions
        buffer_size = 20  # Adjust based on the level of expansion needed

        # Process masks and calculate overlap
        result = process_masks_and_calculate_overlap(mask_path1, mask_path2, buffer_size)
        overlapping_area = result['overlapping_area']
        similarity_matrix.loc[region_images_path[i], region_images_path[j]] = overlapping_area
        similarity_matrix.loc[region_images_path[j], region_images_path[i]] = overlapping_area
    print(similarity_matrix)
    similarity_matrix.to_csv("test.csv")



