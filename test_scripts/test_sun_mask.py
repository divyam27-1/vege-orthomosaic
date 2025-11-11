import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ImageWQItem
import cv2

# Load an image (replace with your actual image path)
image = cv2.imread('/home/iittp/Documents/divyam-btp/other-ortho-softwares/OrthomosaicSLAM/datasets/DJI_20250709102032_0036_D.JPG')

# Assume the position is (lat, lon, alt)
position = (12.34, 56.78, 100)  # Example GPS coordinates

# Create an ImageWQItem instance
image_item = ImageWQItem(image, position, id=1)

# Call the method to display the original and masked images
image_item.show_masked_image()