function plot_times_test(bVitto)
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

x(end+1) = clickTime*numPASCAL;
y(end+1) = clickIOUtest;
t{end+1} = 'Point-level';
b(end+1) = true;

x(end+1) = imgTime*numPASCAL;
y(end+1) = 25.66;
t{end+1} = '[PathakICLR15]';
b(end+1) = false;

x(end+1) = imgTime*numPASCAL;
y(end+1) = 39.6;
t{end+1} = '[Papandreou15]';
b(end+1) = false;

x(end+1) = imgTime*numPASCAL + 20*timePerLabel*60000;
y(end+1) = 40.6;
t{end+1} = '[PinheiroCVPR15]';
b(end+1) = false;

x(end+1) = segmTime*numPASCAL;
y(end+1) = 62.7;
t{end+1} = '[LongCVPR15]';
b(end+1) = false;

x(end+1) = segmTime*numPASCAL;
y(end+1) = 71.6;
t{end+1} = '[ChenICLR15]';
b(end+1) = false;

N = 10*60*60; % measure in tens of hours


plot(x, y,'k.','MarkerSize',markerSize);
hold on;
plot(x(b),y(b),'r.','MarkerSize',markerSizeClick);

for i=1:length(t)
    xx = x(i);
    yy = y(i);
    
    if bVitto
        if b(i)
            yy = yy+5;
            f = fontSizeClick;
        else
            yy = yy - 3.5;      
            f = fontSize;
        end

        h = text(xx,yy,['  ' t{i} ' '],'fontsize',f);
        set(h,'horizontalalignment','center');
        
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
%    xtick = 0:10;
%    axis([0 7 0 80]);
%else
%    xtick = 0:10;
%end
%xticklabel = {};
%for i=1:length(xtick)
%    xticklabel{i} = num2str(2^(xtick(i)));
%end
xlabel('Dataset labeling time (tens of hours)');

%set(gca,'xtick',xtick,'xticklabel',xticklabel);
ylabel('Mean IOU');

if bVitto
    print('-djpeg99','-r300','times_test_vitto.jpg');
else
    print('-djpeg99','-r300','times_test.jpg');
end
