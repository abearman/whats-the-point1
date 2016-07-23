import sys
import os
import os.path
import numpy as np
import PIL
from PIL import Image
import constants

def main(argv):
	image_path = argv[1]
	out_path = argv[2]
	visualize_image(image_path, out_path)


def visualize_image(image_path, out_path):		
	basename = os.path.splitext(os.path.basename(image_path))[0]
	img_arr = np.array(Image.open(image_path))

	labels = np.unique(img_arr).tolist()
	print labels

	output = np.zeros((img_arr.shape[0], img_arr.shape[1], 3))
	for k in labels:
		i, j = np.where(img_arr == k)
		if k == 255:
			output[i, j] = constants.hex_to_rgb("FFFFFF")
		else:
			output[i, j] = constants.hex_to_rgb(constants.COLOR_SCHEME_HEX[k])
	output = output.astype(np.uint8)

	Image.fromarray(output).save(out_path + '/' + basename + '_out.png')


if __name__ == "__main__":
	main(sys.argv)
