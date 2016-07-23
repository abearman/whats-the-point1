import sys
import os
from os.path import isfile, join, splitext

def main(argv):
	IMG_DIR = '/imagenetdb3/abearman/data/pascal-context-59/59_context_labels/'
	#IMG_DIR = '/imagenetdb3/abearman/data/pascal-context-459/trainval/'
	TRAIN_FILE = '/imagenetdb3/abearman/data/pascal-context-59/train.txt'
	VAL_FILE = '/imagenetdb3/abearman/data/pascal-context-59/val.txt'	
	VAL_DIR = '/imagenetdb3/abearman/data/pascal-context-59/59_context_labels/'

	if os.path.exists(TRAIN_FILE):
  	os.remove(TRAIN_FILE)
	if os.path.exists(VAL_FILE):
		os.remove(VAL_FILE)
EWFHO;w
	with open(OUT_FILE, "a") as txt_file:
		for f in os.listdir(IMG_DIR):
			img_prefix = splitext(f)[0]
			if isfile(join(IMG_DIR, f)): 
				txt_file.write(img_prefix + '\n')	

if __name__ == "__main__":
    main(sys.argv)
