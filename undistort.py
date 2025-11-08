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
dist_coeffs = np.array([params["k1"], params["k2"], params["p1"], params["p2"], params["k3"]])  # distortion coefficients

# Camera matrix (3x3)
mtx = np.zeros((3, 3))
mtx[0, 0] = focal_length
mtx[1, 1] = focal_length
mtx[2, 2] = 1.0
mtx[0, 2] = principal_point[0]
mtx[1, 2] = principal_point[1]

def undistort_image(image, cam_matrix, distortion_coeffs, crop_percent=0.0):
    """Undistort a single image."""
    
    h, w = image.shape[:2]
    
    map1, map2 = cv2.initUndistortRectifyMap(cam_matrix, distortion_coeffs, None, cam_matrix, (w, h), cv2.CV_32FC1)
    
    undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    
    if crop_percent > 0:
        h, w = undistorted_image.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        undistorted_image = undistorted_image[crop_h:h - crop_h, crop_w:w - crop_w]
        
    return undistorted_image


def process_and_save_image(image_path, input_dir, output_dir):
    """Helper function to process an image and save it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    undistorted = undistort_image(image, mtx, dist_coeffs, crop_percent=0.0)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, undistorted)

def undistort_images_in_directory(input_dir, output_dir):
    """Process all images in the input directory and save the undistorted images in parallel."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename)) and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_and_save_image, image_path, input_dir, output_dir)
            for image_path in image_paths
        ]
        
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    input_dir = 'datasets/'
    output_dir = 'datasets_undistorted/'

    undistort_images_in_directory(input_dir, output_dir)
    print("All images have been undistorted.")