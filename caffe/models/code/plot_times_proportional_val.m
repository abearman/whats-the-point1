function plot_times_proportional_val(bVitto)
if nargin < 1
    bVitto = true;
end

plot_times;

x = [];
y = [];
 
il_x = [];
il_y = [];

pt_x = [];
pt_y = [];

sq_x = [];
sq_y = [];

box_x = [];
box_y = [];

full_x = [];
full_y = [];

t = {};
%b = logical([]);
c = {};
m = [];
mm = [];
budget_half = 29.40;
budget_hours = 58.79;

il_x(end+1) = 0;
il_y(end+1) = 0;

%il_x(end+1) = budget_half;
%il_y(end+1) = 0;

il_x(end+1) = budget_hours;
il_y(end+1) = 29.8;
x(end+1) = budget_hours;
y(end+1) = 29.8;
t{end+1} = 'Image-level';
c{end+1} = 'blue';
m(end+1) = 3.5;
mm(end+1) = 14.5;

%%

pt_x(end+1) = 0;
pt_y(end+1) = 0;

pt_x(end+1) = budget_half;
pt_y(end+1) = 39.47;

pt_x(end+1) = budget_hours;
pt_y(end+1) = 42.87;
x(end+1) = budget_hours;
y(end+1) = 42.87;
t{end+1} = 'Point-level';
c{end+1} = 'Red';
m(end+1) = -3.5;
mm(end+1) = 15;

%%

sq_x(end+1) = 0;
sq_y(end+1) = 0;

sq_x(end+1) = budget_half;
sq_y(end+1) = 38.61;

sq_x(end+1) = budget_hours;
sq_y(end+1) = 41.14;
x(end+1) = budget_hours;
y(end+1) = 41.14;
t{end+1} = 'Squiggle-level';
c{end+1} = 'magenta';
m(end+1) = 3.5;
mm(end+1) = 17.5;

%%

box_x(end+1) = 0;
box_y(end+1) = 0;

box_x(end+1) = budget_half;
box_y(end+1) = 25.97;

box_x(end+1) = budget_hours;
box_y(end+1) = 23.08;
x(end+1) = budget_hours;
y(end+1) = 23.08;
t{end+1} = 'Box-level';
c{end+1} = 'cyan';
m(end+1) = 4;
mm(end+1) = 12.5;

%%%
full_x(end+1) = 0;
full_y(end+1) = 0;

full_x(end+1) = budget_half;
full_y(end+1) = 20.86;

full_x(end+1) = budget_hours;
full_y(end+1) = 22.07;
x(end+1) = budget_hours;
y(end+1) = 22.07;
t{end+1} = 'Full supervision';
c{end+1} = 'green';
m(end+1) = 2;
mm(end+1) = 18.5;




N = 10*60*60; % measure in tens of hours
plot(il_x, il_y, 'b-', 'Marker', '.', 'MarkerSize', 40);
hold on;
grid on;
plot(pt_x, pt_y, 'r-', 'Marker', '.', 'MarkerSize', 40*1.2);
plot(sq_x, sq_y, 'm-', 'Marker', '.', 'MarkerSize', 40);
plot(box_x, box_y, 'c-', 'Marker', '.', 'MarkerSize', 40);
plot(full_x, full_y, 'g-', 'Marker', '.', 'MarkerSize', 40);




%plot(x(b),y(b),'r.','MarkerSize',markerSizeClick);

for i=1:length(t)
    xx = x(i);
    yy = y(i);
    
    if bVitto
        if strcmp(c{i}, 'Red') == 1
            yy = yy+5;
            f = fontSizeClick;
        else
            yy = yy - 3.5;      
            f = fontSize;
        end
	if length(mm) >= i
            xx = xx + mm(i);
        end
	if length(m) >= i
            yy = yy + m(i);
        end
	

        h = text(xx,yy,['  ' t{i} ' '],'fontsize',f);
        set(h,'horizontalalignment','center');
	
	set(h, 'Color', c{i});

        %if b(i)
         %   set(h,'Color','red');
        %end
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
%    xtick = 0:10;
%    axis([0 7 0 80]);
%else
%    xtick = 0:10;
%end
%xticklabel = {};
%for i=1:length(xtick)
%    xticklabel{i} = num2str(2^(xtick(i)));
%end
xlabel('Annotation time budget (hours)');

%set(gca,'xtick',xtick,'xticklabel',xticklabel);
ylabel('Mean IOU'); 
xlim([0 100]);
ylim([0 60]);

print('-djpeg99','-r300','times_proportional_val.jpg');
