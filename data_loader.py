import os
import re
from PIL import Image

def load_region_images(folder_path):
    region_images = []
    pattern = re.compile(r"region_\d+_with_border\.png")

    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            file_path = os.path.join(folder_path, file_name)
            region_images.append(file_path)
    return region_images


