import json
import os.path
import numpy as np
import shutil
from PIL import Image, ImageDraw
import scipy.misc

PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

IMG_DIR_VOC = "/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/"
IMG_DIR_SBD = "/imagenetdb3/olga/data/PASCAL_SBD/dataset/img/"
#GT_DIR_VOC = "/imagenetdb3/olga/data/VOCdevkit/VOC2012/SegmentationClass/"
#GT_DIR_SBD = "/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls/"

OUT_DIR = "/imagenetdb3/abearman/data/PASCAL_AMT_squiggles/real1_squiggle1/"

filename = "pascal_2012tv_SBDtv_n30_acc80_t628.txt"
output_file = "/afs/cs/u/abearman/whats_the_point_2015/AMT_UI/scripts/results/" + filename

print "Reading results file ..." 
output_json_data = open(output_file).read()
print "Done reading results file"
output_data = json.loads(output_json_data)
print "Done loading results json"

print "There are " + str(len(output_data)) + " HITs to parse."

counter = 1
for hit in output_data:
	print counter
	counter += 1
	img_ids = [str(img["img_id"]) for img in hit["output"]["input"]["questions"]]
	img_outputs = hit["output"]["output"]

	for i in xrange(len(img_outputs)): 
		img_id = img_ids[i]
		img_path_voc = IMG_DIR_VOC + img_id + ".jpg"
		img_path_sbd = IMG_DIR_SBD + img_id + ".jpg"
		img_out_path = OUT_DIR + img_id + ".png"

		original_img = None
		if os.path.isfile(img_path_voc):  # is VOC
			original_img = Image.open(img_path_voc)
		elif os.path.isfile(img_path_sbd):  # is SBD
			original_img = Image.open(img_path_sbd)
		else:
			raise Exception("Couldn't find corresponding image for ID: " + img_id)
		original_img_arr = np.array(original_img)

		visualized_arr = None
		if os.path.isfile(img_out_path): # Already has some pixels labelled
			print "Image already exists!"
			visualized_img = Image.open(img_out_path)
			visualized_arr = np.array(visualized_img)
		else:
			visualized_arr = np.zeros((original_img_arr.shape[0], original_img_arr.shape[1]))
			visualized_arr.fill(255)  # Initialize with all 255's

		click_data = img_outputs[i]
		human_label = hit["output"]["input"]["object_name"]

		for click in click_data["answer"]["clicks"]:
			y = click["y"]		
			x = click["x"]
			if y < visualized_arr.shape[0] and x < visualized_arr.shape[1]:
				visualized_arr[y][x] = PASCAL_CLASSES.index(human_label)
			else:		
				print "Error: out of bounds, not marking squiggle"

		print np.unique(visualized_arr)
		Image.fromarray(np.uint8(visualized_arr)).save(img_out_path)
		
