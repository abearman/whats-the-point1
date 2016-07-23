clicks_dir = '/imagenetdb3/olga/data/PASCAL_AMT_clicks/real1_users2';

numPerClass = str2num(clicks_dir(end))
numPerInstance = 0;

addpath('/afs/cs/u/olga/Research/helper')
fpath = ['/imagenetdb3/abearman/caffe/data/PASCAL_AMT_clicks/real1_users' int2str(numPerClass) '_convex']

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

clicks_dir = '/imagenetdb3/olga/data/PASCAL_AMT_clicks/real1_users3';
%idir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/inst_plus_VOC';
%cdir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/cls_plus_VOC';
%imgdir = '/imagenetdb3/olga/data/PASCAL_SBD/dataset/img';i
d = dir([clicks_dir '/*.png']);
%img_names = dir([imgdir '/*.jpg']);

rng('shuffle');
for i=1:length(d)
	i
    %inst = imread([idir '/' d(i).name]);
    %cls = imread([cdir '/' d(i).name]);
	clicks_im = imread([clicks_dir '/' d(i).name]);

    if numPerClass > 0
        un = unique(clicks_im)';
        un(un == 0 | un == 255) = [];
        %assert(length(un) > 0);
		if length(un) == 0
			'Found image with no clicks on foreground class'
		end
        for k=un
			indices = [];
       		%temp_im = uint8(ones(size(inst))*255);
			temp_im = clicks_im;			
			temp_im(temp_im ~= k) = 255; % Sets all pixels to be either k or 255 (whites out the other classes)

			% Interpolate between user clicks
			temp_im(temp_im == 255) = 0; % Converts to BW
			temp_im(temp_im ~= 0) = 1; % Converts to BW
			ch = bwconvhull(temp_im); % Binary image: 1 for foreground, 0 for background
			temp_im = uint8(ones(size(clicks_im))*255);
			temp_im(ch == 1) = k;

			clicks_im(temp_im == k) = k;
        end

		%unique(clicks_im)
		%imwrite(clicks_im, 'clicks.png');
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

	imwrite(clicks_im,[odir '/' d(i).name]);
end
