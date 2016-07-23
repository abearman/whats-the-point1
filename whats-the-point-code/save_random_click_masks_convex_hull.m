numPerClass = 3;
numPerInstance = 0;

addpath('/afs/cs/u/olga/Research/helper')
fpath = ['/imagenetdb3/abearman/caffe/data/PASCAL_AMT_clicks/random1_users' int2str(numPerClass) '_convex']

i = 1;
%while true
%    if ~exist(sprintf(fpath,i),'dir')
%        mkdir(sprintf(fpath,i));
%        break;
%    end
%    i = i+1;
%end
odir = sprintf(fpath,i);
fprintf('Dir is %s\n',odir);

idir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/inst_plus_VOC';
cdir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC';
imgdir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/img';i
d = dir([idir '/*.png']);
img_names = dir([imgdir '/*.jpg']);

rng('shuffle');
%report(0);
for i=1:length(d)
	i
    %if toc > 10
     %   report(i,length(d));
    %end
    inst = imread([idir '/' d(i).name]);
    cls = imread([cdir '/' d(i).name]);

	%inst = imread([idir '/2010_003151.png']);
	%cls = imread([cdir '/2010_003151.png']);
	%real_img = imread([imgdir '/2010_003151.jpg']);

    im = uint8(ones(size(inst))*255); % Sets all values to be 255

    if numPerClass > 0
        un = unique(cls)';
        un(un == 0 | un == 255) = [];
        assert(length(un) > 0);
        for k=un
			indices = [];
       		temp_im = uint8(ones(size(inst))*255);
		    for j=1:numPerClass
                ind = find(cls == k); % Finds all pixels labelled with this class from the ground truth
                ind = ind(randi(length(ind))); % Chooses a random index
                temp_im(ind) = cls(ind); % Sets the image to have a supervised pixel at this index
				indices(end+1) = ind;
            end
			% Interpolate between random clicks
			temp_im(temp_im == 255) = 0; % Converts to BW
			temp_im(temp_im ~= 0) = 1; % Converts to BW
			ch = bwconvhull(temp_im); % Binary image: 1 for foreground, 0 for background
			temp_im = uint8(ones(size(inst))*255);
			temp_im(ch == 1) = k;

			im(temp_im == k) = k;
        end
		%imwrite(ch, 'ch.png');
		%imwrite(im, 'im.png');
		%imwrite(real_img, 'real_img.jpg');

	%unique(im)
        %assert(sum(sum(im ~= 255)) == numPerClass*length(un));
    else
        un = unique(inst)';
        un(un == 0 | un == 255) = [];
        assert(length(un) > 0);
        for k=un
            for j=1:numPerInstance
                ind = find(inst == k);
                ind = ind(randi(length(ind)));
                im(ind) = cls(ind);
            end
        end
        assert(sum(sum(im ~= 255)) == max(un));
    end
	imwrite(im,[odir '/' d(i).name]);
end
