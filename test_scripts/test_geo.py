import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from main import Orthomosaic
from geo_test_utils import *

ortho = Orthomosaic(debug=True)
ortho.load_dataset(input_dir='datasets/')
images_pool = ortho.images.copy()
clear_directory('steps/')
clear_directory('matched_features/')

ortho.combination_method = 'feather'
ortho.combination_params['feather_width']=60

def get_image_center(img1, img2):
    """Update the positions of the two images to the center of the stitched result."""
    lat_center = (img1.lat + img2.lat) / 2
    lon_center = (img1.lon + img2.lon) / 2
    alt_center = (img1.alt + img2.alt) / 2

    return lat_center, lon_center, alt_center


def recursive_stitching(ortho, images_pool):
    """Recursive stitching of images."""
    
    # Step 1: Find the closest pairs of images in the current pool
    closest_pairs = find_closest_image_pairs(images_pool)

    if not closest_pairs:
        # If no pairs can be stitched, the process is complete.
        print("No more pairs to stitch. Ending process.")
        return images_pool
    
    # Step 2: Get the closest pair and remove it from the pool
    img1, img2, _ = closest_pairs[0]
    images_pool = [img for img in images_pool if img not in [img1, img2]]
    
    # Step 3: Call the stitcher to stitch these two images
    idx_tuple = (img1.id, img2.id)
    status, stitched_image = ortho.sticher(img1.image, img2.image, idx_tuple)
    
    if status:
        # Step 4: Calculate the center for the new stitched image
        center = get_image_center(img1, img2)
        
        # Step 5: Create a new ImageWQItem for the stitched image (use a new dynamic ID)
        new_image = ImageWQItem(stitched_image, center, id=int(str(min(img1.id, img2.id)) + '0'))  # Use dynamic ID
        
        # Step 6: Update the pool:
        images_pool.append(new_image)

        # Step 7: Recurse with the updated pool
        return recursive_stitching(ortho, images_pool)
    else:
        # If stitching fails, skip this pair and try the next
        if img1.image.size > img2.image.size:
            images_pool.append(img1)
        else:
            images_pool.append(img2)

        # Recursively try stitching the next pair
        return recursive_stitching(ortho, images_pool)
    

final_images = recursive_stitching(ortho, images_pool)

# If only one image is left, save the final result
if len(final_images) == 1:
    final_image = final_images[0]
    cv2.imwrite("results/final.jpg", final_image.image)
    print("Final image saved as results/final.jpg")
else:
    print("Error: Multiple images left in the pool.")
