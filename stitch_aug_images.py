import os
import matplotlib.pyplot as plt
from PIL import Image

def stitch_images_in_grid(folder_path, grid_size=(8, 8)):
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    
    # Get all PNG images from the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Ensure we have exactly 64 images
    if len(image_files) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} images, but found {len(image_files)}.")

    # Iterate through the grid and display images
    for ax, img_file in zip(axes.flatten(), image_files):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')  # Use 'gray' for grayscale images
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

# Example usage
stitch_images_in_grid('./augmented_images')