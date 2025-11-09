import sys
import cv2
from PIL import Image
import PIL.ExifTags as ExifTags
import numpy as np
import imutils
from imutils import paths
from undistort import mtx as MTX, dist_coeffs as DCOEFF
from concurrent.futures import ThreadPoolExecutor, as_completed

class Orthomosaic:
    def __init__(self, debug):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.images = []        # list of ImageWQItem
        self.no_raw_images = []
        self.temp_image = []
        self.final_image = []
        self.debug = debug

        self.combination_method = "direct"  # Options: direct, superposition, blend, feather
        self.combination_params = {"alpha": 0.5, "feather_width": 20}
        pass

    def load_dataset(self, input_dir):
        if self.debug:
            print(f"[INFO] Importing Images from directory {input_dir}...")

        imagePaths = sorted(list(paths.list_images(input_dir)))

        def preprocess_image(imagePath, image_id):
            image_temp = cv2.imread(imagePath)
            image_metadata = Image.open(imagePath)._getexif()

            if image_temp is None:
                print(f"[ERROR] Could not read image at {imagePath}")
                return None

            gps_info = None
            if image_metadata is not None:
                for tag, value in image_metadata.items():
                    if ExifTags.TAGS.get(tag) == "GPSInfo":
                        gps_info = value
                        break
            if gps_info is not None:
                lat = gps_info.get(2, (0, 0, 0))
                lon = gps_info.get(4, (0, 0, 0))
                alt = gps_info.get(6, 0)
                pos = (float(lat[0] + lat[1]/60 + lat[2]/3600),
                       float(lon[0] + lon[1]/60 + lon[2]/3600),
                       float(alt))
            else:
                pos = (0.0, 0.0, 0.0)

            image_item = ImageWQItem(image_temp, pos, image_id)
            image_item_img = image_item.undistort(MTX, DCOEFF)
            image_item.image = image_item_img

            image_item_img = image_item.downsample(3)
            image_item.image = image_item_img

            return image_item
        

        with ThreadPoolExecutor() as executor:
            futures = []
            for image_id, imagePath in enumerate(imagePaths):
                futures.append(executor.submit(preprocess_image, imagePath, image_id))

            for future in as_completed(futures):
                image_item = future.result()
                self.images.append(image_item)

        if self.debug:
            print(f"[INFO] Imported {len(self.images)} images.")

    def mixer(self):
        self.no_raw_images = len(self.images)
        if self.debug:
            print(f"[INFO] {self.no_raw_images} Images have been loaded")
        for x in range(self.no_raw_images):
            if x == 0:
                self.temp_image = self.sticher(self.images[x], self.images[x+1], x)
            elif x < self.no_raw_images-1 :
                self.temp_image = self.sticher(self.temp_image, self.images[x+1], x)
            else:
                self.final_image = self.temp_image                

        # self.final_image = self.sticher(self.images[0], self.images[1])
        cv2.imshow("output", self.final_image)
        cv2.imwrite("output.png", self.final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def sticher(self, image1, image2, idx):
        # image1_grayscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image2_grayscale = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=3000)

        # Find the key points and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        all_matches = []
        for m, n in matches:
            all_matches.append(m)

        good = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:
                good.append(m)

        matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'matched_features/matched_{idx[0]}_{idx[1]}.jpg',matched_img)

        # Set minimum match condition
        MIN_MATCH_COUNT = 4

        if len(good) > MIN_MATCH_COUNT:
            # Convert keypoints to an argument for findHomography
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=9.0)
            M_inv = np.linalg.inv(M)

            print(f"[INFO] Stitching images {idx[0]} and {idx[1]} : SUCCESS, {len(good)} matches found")
            result = self.wrap_images(image2, image1, M)
            status = True

        else:
            print(f"[WARN] Stitching images {idx[0]} and {idx[1]} : FAILED, not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}") 
            result = image1
            status = False
        
        cv2.imwrite(f'steps/stitched_{idx[0]}_{idx[1]}.jpg',result)
        return status, result

    def wrap_images(self, base_image, incoming_image, homography):
        # Get dimensions of both images
        base_rows, base_cols = base_image.shape[:2]
        incoming_rows, incoming_cols = incoming_image.shape[:2]

        # Define the corner points of the base image and incoming image
        base_corners = np.float32([[0, 0], [0, base_rows], [base_cols, base_rows], [base_cols, 0]]).reshape(-1, 1, 2)
        incoming_corners = np.float32([[0, 0], [0, incoming_rows], [incoming_cols, incoming_rows], [incoming_cols, 0]]).reshape(-1, 1, 2)

        # Transform the incoming image's corners to the base image's perspective
        transformed_corners = cv2.perspectiveTransform(incoming_corners, homography)

        # Combine the corners of both images to get the overall bounding box
        all_corners = np.concatenate((base_corners, transformed_corners), axis=0)

        # Calculate the new image bounds (min and max coordinates)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        #print(f"Base corners:\n{base_corners}\nTransformed corners:\n{transformed_corners}\n")

        # Calculate the translation required to shift the image into view
        translation = [-x_min, -y_min]

        # Construct the translation matrix
        translation_matrix = np.array([[1, 0, translation[0]], 
                                       [0, 1, translation[1]], 
                                       [0, 0, 1]])

        print(f"[INFO] Output image size: width={x_max - x_min}, height={y_max - y_min}")

        # Warp the incoming image using the combined homography and translation matrix
        warped_image = cv2.warpPerspective(incoming_image, translation_matrix.dot(homography), (x_max - x_min, y_max - y_min))

        # Place the base image into the final output image
        combined_image = self.combine_images(warped_image, base_image, translation, method=self.combination_method, **self.combination_params)

        return combined_image

    def combine_images(self, image_bg, image_fg, translation, method="direct", **kwargs):
        combined_image = image_bg.copy()

        x_offset, y_offset = translation
        fg_rows, fg_cols = image_fg.shape[:2]

        # Place the foreground image onto the background image according to the method
        if method == "direct":
            combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols] = image_fg

        elif method == "superposition":
            alpha = kwargs.get("alpha", 1.5)
            condition = image_fg > combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols] / alpha

            combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols] = np.where(
                condition,
                image_fg,
                combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols]
            )
        
        elif method == "feather":
            # Create a feathered mask
            mask = np.zeros((fg_rows, fg_cols), dtype=np.float32)
            for i in range(fg_rows):
                for j in range(fg_cols):
                    distance_to_edge = min(i, j, fg_rows - i - 1, fg_cols - j - 1)
                    mask[i, j] = min(1.0, distance_to_edge / kwargs.get("feather_width", 50))
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            for c in range(3):  # Assuming 3 color channels
                combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols, c] = (
                    combined_image[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols, c] * (1 - mask) +
                    image_fg[:, :, c] * mask
                )

        return combined_image

class ImageWQItem:
    def __init__(self, image, pos, id):
        self.image = image
        self.lat, self.lon, self.alt = pos
        self.id = id

    def downsample(self, scale_factor=2):
        height, width = self.image.shape[:2]
        new_dimensions = (width // scale_factor, height // scale_factor)
        downsampled_image = cv2.resize(self.image, new_dimensions, interpolation=cv2.INTER_LINEAR)

        return downsampled_image
    
    def undistort(self, cam_matrix, distortion_coeffs, crop_percent=0.0):
        """Undistort a single image using Undistorted Rectify Method"""

        h, w = self.image.shape[:2]

        map1, map2 = cv2.initUndistortRectifyMap(cam_matrix, distortion_coeffs, None, cam_matrix, (w, h), cv2.CV_32FC1)

        undistorted_image = cv2.remap(self.image, map1, map2, cv2.INTER_LINEAR)

        if crop_percent > 0:
            h, w = undistorted_image.shape[:2]
            crop_h = int(h * crop_percent)
            crop_w = int(w * crop_percent)
            undistorted_image = undistorted_image[crop_h:h - crop_h, crop_w:w - crop_w]

        return undistorted_image


if __name__ == "__main__":
    tester = Orthomosaic(debug=True)
    tester.load_dataset()
    tester.combination_method = 'superposition'
    tester.combination_params = {'alpha': 1.5}
    tester.mixer()
else:
    pass
