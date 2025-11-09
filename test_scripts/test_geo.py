import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import Orthomosaic
from geo_test_utils import *

ortho = Orthomosaic(debug=True)
ortho.load_dataset(input_dir='datasets/')
clear_directory('steps/')
clear_directory('matched/')

images_db = {
    image_wq_item.id:image_wq_item
    for image_wq_item in ortho.images
}

closest_pairs = find_closest_image_pairs(ortho.images)

for img1, img2, distance in closest_pairs:    
    idx_tuple = (img1.id, img2.id)
    
    #sticher will automatically save it to steps and matched features directory
    status, stitched_image = ortho.sticher(img1.image, img2.image, idx_tuple)