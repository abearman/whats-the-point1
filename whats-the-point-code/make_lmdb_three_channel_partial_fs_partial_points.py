import sys
sys.path.append('/imagenetdb3/abearman/caffe/python/')
import caffe
import lmdb
from PIL import Image
import numpy as np
import os
import os.path
import time
import scipy 
import random

if len(sys.argv) < 4:
	print 'Usage: python make_lmdb.py <idsfile> <annotfolder> <annotfolder2> <outfolder>'
	exit(-1);

NUM_FS_IMAGES = 100

PASCAL_FOLDER = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/' 
IDS_FILE = sys.argv[1];
DIR = sys.argv[2] + '/';
DIR2 = sys.argv[4] + '/'

OUTDIR = sys.argv[5] + '/';

ids_file = open(IDS_FILE)
ids = [line[:-1] for line in ids_file]

print 'ImageSet:',IDS_FILE
print 'Num:',len(ids)
print 'Folder:',DIR
print 'Folder2:',DIR2
print 'Outfolder:',OUTDIR

if len(ids) == 0:
	print 'Nothing to do'
	exit(0);

if os.path.exists(DIR+ids[0]+'.jpg'):
	bGt = False
elif os.path.exists(DIR+ids[0]+'.png'):
	bGt = True
else:
	print DIR+ids[0]+'.png'
	print 'Is this gt or image folder??'
	exit(-1)

assert(bGt)

in_db = lmdb.open(OUTDIR, map_size=int(1e12))

start_t = time.time()
count = 0

with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(ids):
		if time.time() - start_t > 60:
			count += 1

			# Switch from full to point supervision
			if count == NUM_FS_IMAGES:
				DIR = sys.argv[3] + '/' 		
	
			print in_idx,'/',len(ids),'(',count,'min',')'
			start_t = time.time()

		# load image:
		# - as np.uint8 {0, ..., 255}
		# - in BGR (switch from RGB)
		# - in Channel x Height x Width order (switch from H x W x C)

		print "Image name: " + in_
		im = np.array(Image.open(DIR + '/' + in_ + '.png'),dtype=np.uint8) # or load whatever ndarray you need
		#im[im == 0] = 255;

		if len(im.shape) > 2:
			im = im[:,:,2]
		print im.shape

		(w,h) = im.shape;
		im = im.reshape((1,w,h));

		
		im2 = np.array(Image.open(DIR2 + '/' + in_ + '.png'),dtype=np.uint8) # or load whatever ndarray you need
		if len(im2.shape) > 2:
			im2 = im2[:,:,2]
		(w2,h2) = im2.shape;
		im2 = im2.reshape((1,w2,h2));
		assert(w2 == w)
		assert(h2 == h)

		a = np.unique(im)
		a = a[(a < 255) & (a > 0)]	# gets all the pixels that are object classes
		im3 = np.zeros((1,w,h))
		im3[0,0,0:len(a)] = a

		im4 = np.vstack((im,im2,im3))

		im_dat = caffe.io.array_to_datum(im4)
		in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

os.system('chmod -R 777 ' + OUTDIR)

print '-----'
