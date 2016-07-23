%LMDB_FILE = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012tv_SBDtv.txt';
LMDB_FILE = '/imagenetdb3/abearman/data/pascal-context-59/train.txt';
DATA_PATH = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/';
OUT_PATH = '/imagenetdb3/abearman/data/pascal-objectness/';

fid = fopen(LMDB_FILE);
img_name = fgetl(fid);
i = 0;
while ischar(img_name)
	if ~exist([OUT_PATH img_name '.png'])
		img_name
		img = imread([DATA_PATH img_name '.jpg']);
		windows = runObjectness(img, 1000);
		objHeatMap = computeObjectnessHeatMap(img,windows);
		img_path = [OUT_PATH img_name '.png'];
		imwrite(objHeatMap, img_path)
	else
		[img_name 'already exists']
	end
	img_name = fgetl(fid);
	i
	i = i + 1;
end
fclose(fid);

