import pandas as pd

def process_and_compare_masks(mask_filepath, sim_filepath, test_filepath, sim_threshold=100):
    """
    Process and compare masks based on similarity matrices:
    1. Filter similar masks from sim.csv based on sim_threshold.
    2. Compare the query mask directly with the filtered masks in test.csv.
    
    Args:
        mask_filepath (str): The filepath of the target mask.
        sim_filepath (str): The path to the sim.csv file.
        test_filepath (str): The path to the test.csv file.
        sim_threshold (float): Minimum similarity score for sim.csv.
    """
    # Load the similarity matrices
    print("Loading similarity matrices...")
    sim_matrix = pd.read_csv(sim_filepath, index_col=0)
    test_matrix = pd.read_csv(test_filepath, index_col=0)

    # Check if the given mask exists in sim.csv
    if mask_filepath not in sim_matrix.index:
        print(f"Error: Mask '{mask_filepath}' not found in sim.csv.")
        return

    print(f"\nStep 1: Filtering similar masks in sim.csv for mask '{mask_filepath}' with threshold {sim_threshold}...")
    
    # Get all masks that are similar in sim.csv
    similar_masks_sim = sim_matrix.loc[mask_filepath]
    similar_masks_sim = similar_masks_sim[similar_masks_sim > sim_threshold]

    if similar_masks_sim.empty:
        print(f"No masks in sim.csv have similarity greater than {sim_threshold} with '{mask_filepath}'.")
        return

    print(f"Found {len(similar_masks_sim)} similar masks in sim.csv with their scores:")
    print(similar_masks_sim)

    filtered_mask = similar_masks_sim.index[0]  # Taking the first item

    print(f"\nStep 2: Comparing the query mask '{mask_filepath}' with the filtered mask '{filtered_mask}' in test.csv...")
    
    # Check if the query mask and filtered mask exist in test.csv
    if mask_filepath in test_matrix.index and filtered_mask in test_matrix.columns:
        score = test_matrix.at[mask_filepath, filtered_mask]
        print(f"Similarity score in test.csv between '{mask_filepath}' and '{filtered_mask}': {score}")
    else:
        print(f"Either '{mask_filepath}' or '{filtered_mask}' is not present in test.csv.")

# Usage
mask_filepath = "data/row_0_col_1638/region_12_with_border.png"  # Replace with the actual mask filepath
sim_filepath = "sim.csv"  
test_filepath = "test.csv"  

print("Starting the mask similarity filtering process...")
process_and_compare_masks(mask_filepath, sim_filepath, test_filepath, sim_threshold=100)
print("Process complete!")
