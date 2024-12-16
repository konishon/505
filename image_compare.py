import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from torch.nn.functional import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_region_images
from compare import process_masks_and_calculate_overlap
import pandas as pd
from tqdm import tqdm



def load_image(image_path):
    """Load an image and preprocess it."""
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Keep the image size as 1024x1024
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def load_mask(path, target_shape=None):
    """Load a binary mask, convert to 0 or 1, and optionally resize to target shape."""
    mask = Image.open(path).convert("L")
    if target_shape:
        mask = mask.resize(target_shape, Image.NEAREST)
    return np.array(mask) // 255

def clip_image_with_mask(image, mask):
    """Apply a binary mask to an image and crop to minimize black pixels."""
    clipped = image * mask[:, :, np.newaxis]
    # Find bounding box of the mask
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # Add 1 to include the max pixel
        clipped = clipped[y0:y1, x0:x1, :]
    return clipped

def resize_clipped_image(clipped_image, target_size=(224, 224)):
    """Resize the clipped image to a fixed target size."""
    clipped_image_pil = Image.fromarray((clipped_image * 255).astype(np.uint8))
    resized_image = clipped_image_pil.resize(target_size, Image.BILINEAR)
    return np.array(resized_image) / 255.0

def plot_images(images, titles, figsize=(16, 8)):
    """Plot a list of images with corresponding titles."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def get_embeddings_from_clipped(clipped_image, model):
    """Generate embeddings for a clipped image using a pre-trained model."""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Convert numpy array back to PIL Image
    clipped_image_pil = Image.fromarray((clipped_image * 255).astype(np.uint8))
    image_tensor = preprocess(clipped_image_pil).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embeddings = model(image_tensor)
    return embeddings.squeeze()  # Remove batch dimension

def cosine_similarity_check_clipped(clipped_image1, clipped_image2):
    """Compute cosine similarity between embeddings of two clipped images."""
    # Load pre-trained ResNet50 model
    base_model = resnet50(pretrained=True)
    model = nn.Sequential(*list(base_model.children())[:-1])  # Remove the classification head
    model.eval()  # Set the model to evaluation mode

    # Get embeddings
    embedding1 = get_embeddings_from_clipped(clipped_image1, model)
    embedding2 = get_embeddings_from_clipped(clipped_image2, model)

    # Compute cosine similarity
    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def main():
    # File paths
    image_path = "data/row_0_col_1638/original_image.png"  # Replace with the path to your image
    mask1_path = "data/row_0_col_1638/region_10_with_border.png"  # Replace with the path to your first binary mask
    mask2_path = "data/row_0_col_1638/region_11_with_border.png"  # Replace with the path to your second binary mask
    # Flag for resizing masks to match image size
    resize_masks = True

    # Load data
    image = np.array(Image.open(image_path)) / 255.0
    target_shape = (image.shape[1], image.shape[0]) if resize_masks else None  # Target shape for masks (width, height)
    mask1 = load_mask(mask1_path, target_shape)
    mask2 = load_mask(mask2_path, target_shape)

    # Combine masks
    combined_mask = mask1 & mask2

    # Clip images
    clipped_image1 = clip_image_with_mask(image, mask1)
    clipped_image2 = clip_image_with_mask(image, mask2)
    clipped_combined = clip_image_with_mask(image, combined_mask)

    # Resize clipped images to ensure consistent dimensions
    resized_clipped_image1 = resize_clipped_image(clipped_image1)
    resized_clipped_image2 = resize_clipped_image(clipped_image2)

    # Plot results
    plot_images(
        [image, resized_clipped_image1, resized_clipped_image2, clipped_combined],
        ["Original Image", "Clipped with Mask 1", "Clipped with Mask 2", "Clipped with Combined Mask"]
    )

    # Example usage of cosine similarity
    similarity = cosine_similarity_check_clipped(resized_clipped_image1, resized_clipped_image2)
    print(f"Cosine Similarity: {similarity}")
    


# def main2():    
#     image_path = "data/row_0_col_1638/original_image.png"
#     folder_path = "data/row_0_col_1638"
#     region_images_path = load_region_images(folder_path)
    
#     similarity_matrix = pd.DataFrame(0, index=region_images_path, columns=region_images_path)
#     for i, mask_path1 in tqdm(enumerate(region_images_path), desc="Processing Regions"):
#         for j, mask_path2 in enumerate(region_images_path[i:], start=i):
#             if mask_path1 == mask_path2:
#                 continue  # Skip self-comparison
#             resize_masks = True

#             # Load data
#             image = np.array(Image.open(image_path)) / 255.0
#             target_shape = (image.shape[1], image.shape[0]) if resize_masks else None  # Target shape for masks (width, height)
#             mask1 = load_mask(mask_path1, target_shape)
#             mask2 = load_mask(mask_path2, target_shape)

#             # Combine masks
#             combined_mask = mask1 & mask2

#             # Clip images
#             clipped_image1 = clip_image_with_mask(image, mask1)
#             clipped_image2 = clip_image_with_mask(image, mask2)
#             clipped_combined = clip_image_with_mask(image, combined_mask)

#             # Resize clipped images to ensure consistent dimensions
#             resized_clipped_image1 = resize_clipped_image(clipped_image1)
#             resized_clipped_image2 = resize_clipped_image(clipped_image2)

#             # # # Plot results
#             # plot_images(
#             #     [image, resized_clipped_image1, resized_clipped_image2, clipped_combined],
#             #     ["Original Image", "Clipped with Mask 1", "Clipped with Mask 2", "Clipped with Combined Mask"]
#             # )

#             # Example usage of cosine similarity
#             similarity = cosine_similarity_check_clipped(resized_clipped_image1, resized_clipped_image2)
#             similarity_matrix.loc[region_images_path[i], region_images_path[j]] = similarity
#             similarity_matrix.loc[region_images_path[j], region_images_path[i]] = similarity
#     print(similarity_matrix)
#     similarity_matrix.to_csv("test.csv")
if __name__ == "__main__":
    main()
