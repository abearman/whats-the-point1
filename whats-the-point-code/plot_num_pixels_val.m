function plot_times_val(bVitto)

if nargin < 1
    bVitto = true;
end

plot_times;

x = [];
y = [];
t = {};
b = logical([]);
m = [];
mm = [];



x(end+1) = 0;
y(end+1) = imgIOU;
t{end+1} = 'Image-level';
b(end+1) = false;
m(end+1) = 0.75;
mm(end+1) = 5.0;

x(end+1) = 1.5;
y(end+1) = clickIOU;
t{end+1} = 'Point-level';
b(end+1) = true;
m(end+1) = 0.75;
mm(end+1) = -0.1;

x(end+1) = 502.7;
y(end+1) = squiggleIOU;
t{end+1} = 'Squiggle-level';
b(end+1) = false;
m(end+1) = 0.75;
mm(end+1) = -1.0;

%x(end+1) = 179141.790292;
%y(end+1) = fsIOU;
%t{end+1} = 'Full supervision';
%b(end+1) = false;
%m(end+1) = 0.75;
%mm(end+1) = -0.2;

N = 1; % measure in seconds

plot(x,y,'k.','MarkerSize',markerSize);
hold on;
grid on;
plot(x(b),y(b),'r.','MarkerSize',markerSizeClick);

for i=1:length(t)
    xx = x(i);		
    %xx = log(x(i)/N);
    yy = y(i);
    
    if bVitto
        if length(mm) >= i
            xx = xx + mm(i);
        end
        if length(m) >= i 
            yy = yy + m(i);
        end
        if b(i)
            f = fontSizeClick;
        else
            f = fontSize;
        end

        h = text(xx,yy,['  ' t{i} ' '],'fontsize',f);
        %set(h,'horizontalalignment','center');
        set(h,'horizontalalignment','right');
        if (strcmp(t{i}, 'Squiggle-level') == 0) 
		set(h, 'horizontalalignment', 'left')
	else
		set(h, 'verticalalignment', 'top')
	end
	if b(i)
            set(h,'Color','red');
        end
    else
        h = text(xx,yy,['  ' t{i} '  '],'fontsize',fontSize);
        if x(i) == max(x)
            set(h,'horizontalalignment','right');
        end
        if b(i)
            set(h,'Color','red');
        end
    end
end
hold off;

set(gca,'fontsize',fontSize);    

%if bVitto
%    axis([0 9.5 0 65]);
%    xtick = 0:2:10;
%else
%    axis([4 8 30 60]);
%    xtick = 0:1:10;
%end
%xticklabel = {};
%for i=1:length(xtick)
%    xticklabel{i} = num2str(2^xtick(i));
%end
%}

xlabel('Number of pixels annotated per image');

%set(gca,'xtick',xtick,'xticklabel',xticklabel);
ylabel('Mean IOU');

%print('-djpeg99','-r300','num_pixels_val.jpg');
