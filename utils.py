import os
import shutil
import cv2
import numpy as np

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

def crop_black_borders(image, threshold=10):
    """
    Crops black borders from an image.
    
    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        threshold (int): Pixel intensity threshold to consider as black.
                         Default is 10 (out of 255).
    
    Returns:
        np.ndarray: Cropped image without black borders.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create a binary mask where non-black pixels are white
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find all non-zero (non-black) points
    coords = cv2.findNonZero(mask)
    if coords is None:
        # Entire image is black
        return image

    # Get bounding box of non-black area
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the original image to the bounding box
    cropped = image[y:y+h, x:x+w]

    return cropped
