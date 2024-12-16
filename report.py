import pandas as pd

# Load the similarity matrices
similarity_matrix_1 = pd.read_csv('sim.csv', index_col=0)
similarity_matrix_2 = pd.read_csv('test.csv', index_col=0)

# Define a function to query key relationships
def get_key_relationships(matrix, threshold=0):
    """
    Extract key relationships from a similarity matrix based on a given threshold.
    
    Args:
        matrix (pd.DataFrame): The similarity matrix.
        threshold (float): The minimum similarity value to consider.
        
    Returns:
        pd.DataFrame: DataFrame containing key relationships.
    """
    # Convert the matrix into a long-format DataFrame
    relationships = (
        matrix.stack()  # Stacks the matrix into a multi-index Series
        .reset_index()  # Converts it into a DataFrame
        .rename(columns={"level_0": "Region1", "level_1": "Region2", 0: "Similarity"})  # Rename columns
    )
    
    # Filter relationships based on the threshold and exclude self-similarities
    filtered_relationships = relationships[
        (relationships["Similarity"] > threshold) & (relationships["Region1"] != relationships["Region2"])
    ].sort_values(by="Similarity", ascending=False)
    
    return filtered_relationships

# Set thresholds for both matrices
threshold_matrix_1 = 100  
threshold_matrix_2 = 0.8 

# Query key relationships for both matrices
key_relationships_sim = get_key_relationships(similarity_matrix_1, threshold=threshold_matrix_1)
key_relationships_test = get_key_relationships(similarity_matrix_2, threshold=threshold_matrix_2)

# Save and print results
key_relationships_sim.to_csv('key_relationships_sim.csv', index=False)
key_relationships_test.to_csv('key_relationships_test.csv', index=False)

print("Key Relationships from sim.csv:")
print(key_relationships_sim)
print("\nKey Relationships from test.csv:")
print(key_relationships_test)
