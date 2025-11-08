import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
with open('dewarp_param.yaml') as file:
    params = yaml.safe_load(file)

# Camera parameters (from your EXIF data)
focal_length = params['fx']
principal_point = (params['H']/2 - params['cx'], params['V']/2 + params['cy'])
dist_coeffs = np.array([params["k1"], params["k2"], params["p1"], params["p2"]])  # distortion coefficients

# Camera matrix (3x3)
mtx = np.array([[focal_length, 0, principal_point[0]],
                [0, focal_length, principal_point[1]],
                [0, 0, 1]])

def undistort_image(image, cam_matrix, distortion_coeffs, crop_percent):
    """Undistort a single image."""
    
    # Get the image size
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix (to prevent clipping)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion_coeffs, (w, h), 1, (w, h))

    # Undistort the image using the fisheye model
    undistorted_image = cv2.fisheye.undistortImage(image, cam_matrix, distortion_coeffs, None, new_mtx)

    # Crop a small percentage from the image (optional, as per your initial request)
    crop_pixels = undistorted_image.shape[0] * crop_percent
    undistorted_image = undistorted_image[
        int(crop_pixels):int(undistorted_image.shape[0]-crop_pixels),
        int(crop_pixels):int(undistorted_image.shape[1]-crop_pixels)
    ]

    return undistorted_image


def process_and_save_image(image_path, input_dir, output_dir):
    """Helper function to process an image and save it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    undistorted = undistort_image(image, mtx, dist_coeffs, crop_percent=0.2)
    
    # Get the filename and save the undistorted image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, undistorted)

def undistort_images_in_directory(input_dir, output_dir):
    """Process all images in the input directory and save the undistorted images in parallel."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all the image paths
    image_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename)) and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    # Create a ThreadPoolExecutor to process the images in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_and_save_image, image_path, input_dir, output_dir)
            for image_path in image_paths
        ]
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # Retrieve the result of the future (if needed)

if __name__ == "__main__":
    # Directory paths
    input_dir = 'datasets/'
    output_dir = 'datasets_undistorted/'

    # Undistort all images in the input directory
    undistort_images_in_directory(input_dir, output_dir)
    print("All images have been undistorted.")