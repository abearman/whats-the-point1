addpath('GraphSeg/');
addpath('GraphSeg/coherenceFilter/');
addpath('GraphSeg/GLtree3DMex/');

%% Compile
fprintf('COMPILING:\n')
mex 'GraphSeg/GraphSeg_mex.cpp'
fprintf('\tGraphSeg_mex.cpp: mex succesfully completed.\n')

mex 'GraphSeg/GLtree3DMex/BuildGLTree.cpp'
fprintf('\tBuildGLTree : mex succesfully completed.\n')

mex 'GraphSeg/GLtree3DMex/KNNSearch.cpp'
fprintf('\tKNNSearch : mex succesfully completed.\n')

mex 'GraphSeg/GLtree3DMex/DeleteGLTree.cpp'
fprintf('\tDeleteGLTree : mex succesfully completed.\n\n')


img_dir_pascal = '/imagenetdb3/olga/data/VOCdevkit/VOC2012/JPEGImages/';
img_dir_sbd = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC/';
out_path = '/imagenetdb3/abearman/data/pascal-oversegmented/';
image_ids_file = '/imagenetdb3/olga/data/segm_lmdb/pascal_2012tv_SBDtv.txt';

fid = fopen(image_ids_file);

img_id = fgetl(fid);
while ischar(img_id)	
	% If we haven't already done this image
	if exist([out_path img_id '.png'], 'file') == 0
		img_path = [img_dir_pascal img_id '.jpg'];
	  if exist(img_path, 'file') == 0
  	  img_path = [img_dir_dbd img_id '.jpg'];
  	end   
  	img = imread(img_path);
  	gray_img = rgb2gray(img);
  	filted_I = CoherenceFilter(gray_img, struct('T', 5, 'rho', 2, 'Scheme', 'I', 'sigma', 1));
  	%L = graphSeg(filted_I, 0.5/sqrt(3), 50, 10, 1);
  	L = graphSeg(filted_I, 0.5, 50, 2, 0);
		unique(L)
  	%L = label2rgb(L);
  	fprintf('Finished oversegmentation');
  	imwrite(L, [out_path img_id '.png']);
		end

	% Next iteration
	img_id = fgetl(fid);
end

fclose(fid);

%function L = graphSeg(img, threshold, min_size, nRadius, model)
%Input:
%       img: the gray image
%       threshold: larger prefer larger segmented area
%       min_size: the minimum size of segmentation component
%       nRadius: the radius of neighbourhood of a pixel if in model (0)
%       nRadius: the number of nearest neighborhood of a pixel if in model
%       (1)
%       model: 0-->adjacent neighborhood based
%              1-->k nearest neighborhood based
%       Note: the precisely meaning of above parameters please refer to [1]
%Output:
%       L: the labeled image, differente area is labeled by different
%       number
%Examples:
% %load an gray image:
% load clown;
% I_gray = X;
% %smooth the image by coherence filter:
% filted_I = CoherenceFilter(I_gray,struct('T',5,'rho',2,'Scheme','I', 'sigma', 1));
% %adjacent neighborhood  model:
% L = graphSeg(filted_I, 0.5, 50, 2, 0);
% %k-nearest neighborhood model:
% Lnn = graphSeg(filted_I, 0.5/sqrt(3), 50, 10, 1);
% %display:
% subplot(3, 1, 1), imshow(I_gray, []), title('original image');
% subplot(3, 1, 2), imshow(label2rgb(L)), title('adjacent neighborhood based segmentation');
% subplot(3, 1, 3), imshow(label2rgb(Lnn)), title('k nearest neighborhood based segmentation');
