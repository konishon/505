
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from data_loader import load_region_images
from compare import process_masks_and_calculate_overlap
from PIL import Image
from image_compare import cosine_similarity_check_clipped,clip_image_with_mask,load_mask,resize_clipped_image
from shape_compact import calculate_compactness_ratio


folder_path = "data/row_0_col_1638"
image_path = folder_path + "/original_image.png" 
region_images_path = load_region_images(folder_path)
proximity_similarity_matrix = pd.DataFrame(0, index=region_images_path, columns=region_images_path)
image_similarity_matrix = pd.DataFrame(0, index=region_images_path, columns=region_images_path)
buffer_size = 20 
resize_mask = True
results = []


assert region_images_path is not None
assert len(region_images_path) > 0

for i, mask_path1 in tqdm(enumerate(region_images_path), desc="Processing Regions"):
    for j, mask_path2 in enumerate(region_images_path[i:], start=i):
        if mask_path1 == mask_path2:
            continue      
        # Proximity Analysis 
        result = process_masks_and_calculate_overlap(mask_path1, mask_path2, buffer_size)
        overlapping_area = result['overlapping_area']
        proximity_similarity_matrix.loc[region_images_path[i], region_images_path[j]] = overlapping_area
        proximity_similarity_matrix.loc[region_images_path[j], region_images_path[i]] = overlapping_area

for i, mask_path1 in tqdm(enumerate(region_images_path), desc="Processing Regions"):
     
    proximity_similarity_threshold = 0
    proximity_similarity_table = proximity_similarity_matrix.loc[mask_path1]
    proximity_similarity_table = proximity_similarity_table[proximity_similarity_table > proximity_similarity_threshold]
    
    
    for mask_path2, proximity_score in proximity_similarity_table.items():
        #print(f"Index: {mask_path2}, Value: {proximity_score}")  
        # Image Embeddings Similarity 
        resize_masks = True
        image = np.array(Image.open(image_path)) / 255.0
        target_shape = (image.shape[1], image.shape[0]) if resize_masks else None  # Target shape for masks (width, height)
        mask1 = load_mask(mask_path1, target_shape)
        mask2 = load_mask(mask_path2, target_shape)
        # Combine masks
        combined_mask = mask1 & mask2
        # Clip images
        clipped_image1 = clip_image_with_mask(image, mask1)
        clipped_image2 = clip_image_with_mask(image, mask2)
        clipped_combined = clip_image_with_mask(image, combined_mask)
        # Resize clipped images to ensure consistent dimensions
        resized_clipped_image1 = resize_clipped_image(clipped_image1)
        resized_clipped_image2 = resize_clipped_image(clipped_image2)
        image_similarity = cosine_similarity_check_clipped(resized_clipped_image1, resized_clipped_image2)
    
        compactness_ratio, combined_mask = calculate_compactness_ratio([mask1,mask2])
        
        # Append results
        results.append({
            "mask_path1": mask_path1,
            "mask_path2": mask_path2,
            "proximity": proximity_score,
            "image_similarity": image_similarity,
            "compactness": compactness_ratio
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv")

# Normalize each column
results_df['proximity_norm'] = results_df['proximity'] / results_df['proximity'].max()
results_df['compactness_norm'] = 1 - (results_df['compactness'] / results_df['compactness'].max())
results_df['image_similarity_norm'] = results_df['image_similarity'] / results_df['image_similarity'].max()

# Define weights
w_proximity = 0.4
w_compactness = 0.4
w_image_similarity = 0.2

# Calculate weighted score
results_df['weighted_score'] = (
    w_proximity * results_df['proximity_norm'] +
    w_compactness * results_df['compactness_norm'] +
    w_image_similarity * results_df['image_similarity_norm']
)

# Sort by weighted score in descending order
sorted_results = results_df.sort_values(by='weighted_score', ascending=False)
print(sorted_results[['mask_path1', 'mask_path2', 'weighted_score']])

sorted_results.to_csv("highscrore.csv")