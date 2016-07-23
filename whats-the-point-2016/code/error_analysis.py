# Error checking and timing for a given set of AMT annotations, vs. the ground truth.

from PIL import Image 
import numpy as np

#ANNOTATIONS_DIR = "/imagenetdb3/abearman/data/PASCAL_AMT_squiggles/real1_squiggle1/"
#ANNOTATIONS_DIR = "/imagenetdb3/olga/data/PASCAL_bbox_masks/"
ANNOTATIONS_DIR = "/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC/"
GT_DIR = "/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC/" 

# Image ids for PASCAL VOC train/val + PASCAL SBD train/val
IDS_FILE = "/imagenetdb3/olga/data/segm_lmdb/pascal_2012tv_SBDtv.txt"

times_map = {}
with open('/imagenetdb3/abearman/whats-the-point-code/PASCAL_AMT_TIMES.txt') as times_file:
	for line in times_file:
		arr = line.split()
		img_id = arr[0]		
		time = arr[1]
		times_map[img_id] = float(time)

avg_num_clicks_per_image = 0.0
percent_images_incorrectly_labelled_absent = 0.0
percent_incorrect_clicks = 0.0
avg_time_per_image = 0.0
percent_difficult_clicks = 0.0

num_images = sum(1 for line in open(IDS_FILE))
i = 1
with open(IDS_FILE) as ids_file:
	for image_id in ids_file:  # Strips the newlines
		print i
		i += 1
		image_id = image_id[:-1]
		gt_path = GT_DIR + image_id + '.png'
		annot_path = ANNOTATIONS_DIR + image_id + '.png'
		annot_img = np.array(Image.open(annot_path))
		gt_img = np.array(Image.open(gt_path))

		# Average number of clicks per image
		num_points = (annot_img != 255).sum()
		avg_num_clicks_per_image += num_points 

		# Percentage of images where the object class was incorrectly labelled as absent
		num_non_bg_non_difficult_points = ((gt_img != 255) & (gt_img != 0)).sum() 
		if (num_points == 0) and (num_non_bg_non_difficult_points != 0):
			percent_images_incorrectly_labelled_absent += 1

		# Number of incorrectly labelled pixels 
		#num_mismatched_pixels = ((annot_img != 255) & (annot_img != gt_img)).sum()
		print "gt unique: " + str(np.unique(gt_img))
		print "annot unique: " + str(np.unique(annot_img))

		num_mismatched_pixels = ((annot_img != gt_img) & ((annot_img != 255) | (gt_img == 255)) & ((annot_img != 255) | (gt_img != 0)) ).sum() 
		num_clicks_on_difficult = ((gt_img == 255) & (annot_img != 255) & (annot_img != 0)).sum() 

		if num_points > 0:
			percent_incorrect_clicks += ((float)(num_mismatched_pixels) / num_points)
			percent_difficult_clicks += ((float)(num_clicks_on_difficult) / num_points)

		# Average time for squiggling on an image
		avg_time_per_image += times_map[image_id]

avg_num_clicks_per_image /= num_images

percent_images_incorrectly_labelled_absent /= num_images
percent_images_incorrectly_labelled_absent *= 100

percent_incorrect_clicks /= num_images
percent_incorrect_clicks *= 100

percent_difficult_clicks /= num_images
percent_difficult_clicks *= 100

avg_time_per_image /= num_images

print "avg num clicks per image: " + str(avg_num_clicks_per_image)
print "percent incorrectly labelled absent: " + str(percent_images_incorrectly_labelled_absent)
print "percent incorrect clicks: " + str(percent_incorrect_clicks)
print "percent difficult incorrect: " + str(percent_difficult_clicks)
print "avg time per image: " + str(avg_time_per_image)
