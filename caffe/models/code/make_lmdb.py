import sys
sys.path.append('/imagenetdb3/abearman/caffe/python/')
import caffe
import lmdb
from PIL import Image
import numpy as np
import os
import os.path

if len(sys.argv) < 4:
	print 'Usage: python make_lmdb.py <idsfile> <annotfolder> <outfolder>'
	exit(-1);

PASCAL_FOLDER = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/' 
IDS_FILE = sys.argv[1];
if IDS_FILE == 't':
	IDS_FILE = PASCAL_FOLDER + 'ImageSets/Segmentation/train_2011.txt'
elif IDS_FILE == 'v':
	IDS_FILE = PASCAL_FOLDER + 'ImageSets/Segmentation/val_2011.txt'

DIR = sys.argv[2];
if DIR == 'i':
	DIR = PASCAL_FOLDER + 'JPEGImages/'
elif DIR == 'g':
	 DIR = PASCAL_FOLDER + 'SegmentationClass/'

OUTDIR = sys.argv[3];

ids_file = open(IDS_FILE)
ids = [line[:-1] for line in ids_file]

print 'ImageSet:',IDS_FILE
print 'Num:',len(ids)
print 'Folder:',DIR
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

print 'Gt:',bGt

try:
    os.stat(OUTDIR)
except:
    os.makedirs(OUTDIR)

in_db = lmdb.open(OUTDIR, map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(ids):
		# load image:
		# - as np.uint8 {0, ..., 255}
		# - in BGR (switch from RGB)
		# - in Channel x Height x Width order (switch from H x W x C)

		if bGt:
			im = np.array(Image.open(DIR + '/' + in_ + '.png'),dtype=np.uint8) # or load whatever ndarray you need
			#im[im == 0] = 255;
			(w,h) = im.shape;
			im = im.reshape((1,w,h));
		else:
			im = np.array(Image.open(DIR + '/' + in_ + '.jpg'),dtype=np.uint8) # or load whatever ndarray you need
			im = im[:,:,::-1]
			im = im.transpose(2,0,1)

                im_dat = caffe.io.array_to_datum(im)
		in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

os.system('chmod -R 777 ' + OUTDIR)

print '-----'
