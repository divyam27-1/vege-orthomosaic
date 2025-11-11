import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from main import Orthomosaic

img_1 = cv2.imread('/home/iittp/Documents/divyam-btp/other-ortho-softwares/OrthomosaicSLAM/steps/stitched_570_590.jpg')
img_2 = cv2.imread('/home/iittp/Documents/divyam-btp/other-ortho-softwares/OrthomosaicSLAM/steps/stitched_2000_3000.jpg')

ortho = Orthomosaic(debug=False)

ortho.sticher(img_1, img_2, (99,99))