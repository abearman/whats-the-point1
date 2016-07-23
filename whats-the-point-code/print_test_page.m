function print_results_page()

ids = textread('/imagenetdb3/olga/data/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt', '%s');

folder = '/imagenetdb/www/home/internal/point_project/viz/pascal_test';
base_folder = '//image-net.org/internal/point_project/viz/pascal_test/';

real1_click1_cls_con_obj = [base_folder 'real1-click1-cls-con-obj'];

Nperpage = 100;
Npages = ceil(length(ids)/Nperpage);

fid = fopen([folder '/index.html'],'w');
fprintf(fid,'Train<br>\n');
for i=1:Npages
    fprintf(fid,'<a href="page%i.html">Page%i</a><br>\n', i, i);
end
fclose(fid);

for i=1:Npages
    fid = fopen(sprintf('%s/page%i.html',folder,i),'w');
    start_k = (i-1)*Nperpage+1;
    end_k = min(i*Nperpage,length(ids));
    for k=start_k:end_k
	fprintf(fid, '<table>\n<tr align=\"center\">\n');
	fprintf(fid, '<td>%s</td>\n<td>IMG + OBJ + 1 click</td>\n</tr>\n', ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_in.png>\n</td>\n', real1_click1_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', real1_click1_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img height=120 src=legend11.png>\n</td>\n');
	fprintf(fid,'<td>\n<img height=120 src=legend22.png>\n</td>\n</tr>\n</table>\n');
    end
    fclose(fid);
end
