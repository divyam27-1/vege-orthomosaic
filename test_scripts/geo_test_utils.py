import shutil
import os
import math
from typing import List, Tuple
from main import ImageWQItem

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two geographical points."""
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def find_closest_image_pairs(images: List[ImageWQItem]) -> List[Tuple[ImageWQItem, ImageWQItem, float]]:
    """Find the closest pairs of images based on geographic coordinates."""
    closest_pairs = []
    remaining_images = images.copy()  # Copy of the images list for pairing
    
    while remaining_images:
        image_item_1 = remaining_images.pop(0)  # Take the first image from the list
        closest_distance = float('inf')
        closest_image = None

        # Compare with every other image in the remaining images to find the closest one
        for image_item_2 in remaining_images:
            # Calculate the distance between the two images
            distance = haversine_distance(
                image_item_1.lat, image_item_1.lon,
                image_item_2.lat, image_item_2.lon
            )

            # Update closest image if this one is closer
            if distance < closest_distance:
                closest_distance = distance
                closest_image = image_item_2

        # If a closest image was found, create the pair and remove both from remaining images
        if closest_image:
            closest_pairs.append((image_item_1, closest_image, closest_distance))
            remaining_images.remove(closest_image)  # Remove the paired image from the list

    return closest_pairs


def clear_directory(directory_path):
    """Remove all files and subdirectories from the specified directory."""
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectory and all its contents
                else:
                    os.remove(file_path)  # Remove file
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"Directory {directory_path} cleared.")
    else:
        print(f"Directory {directory_path} does not exist.")

# Clear the 'steps/' directory before running your logic
clear_directory('steps/')
