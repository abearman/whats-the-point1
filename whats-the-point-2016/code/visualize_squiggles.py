import os
import os.path
import shutil
from PIL import Image, ImageDraw, ImageEnhance
import constants
import numpy as np


ANNOT_DIR = "/imagenetdb3/abearman/data/PASCAL_AMT_squiggles/real1_squiggle1/annotated_images/"
VIZ_DIR = "/imagenetdb3/abearman/data/PASCAL_AMT_squiggles/real1_squiggle1/visualized_images/"
IMG_DIR = "/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/"

# Only going to visualize the VOC images, not SBD
for annot_img_file in os.listdir(ANNOT_DIR):
	print annot_img_file
	IMG_ID = os.path.splitext(annot_img_file)[0]	
	shutil.copy2(IMG_DIR + IMG_ID + ".jpg", VIZ_DIR + IMG_ID + ".png")
	visualized_img = Image.open(VIZ_DIR + IMG_ID + ".png")
	bright = ImageEnhance.Brightness(visualized_img)
	visualized_img = bright.enhance(0.5)
	viz_img_arr = np.array(visualized_img)
	annot_img_arr = np.array(Image.open(ANNOT_DIR + IMG_ID + ".png"))

	draw = ImageDraw.Draw(visualized_img)
	rad = 3
	for r in range(viz_img_arr.shape[0]):
		for c in range(viz_img_arr.shape[1]):
			k = annot_img_arr[r, c]
			if k != 255:
				draw.ellipse((c-rad, r-rad, c+rad, r+rad), fill=constants.COLOR_SCHEME_HEX[k])
	visualized_img.save(VIZ_DIR + IMG_ID + ".png")


