function print_val_page()

imgset = '2012v';

idspath = '/imagenetdb3/olga/data/segm_lmdb/pascal_%s.txt';
ids = textread(sprintf(idspath,imgset),'%s');

folder = '/imagenetdb/www/home/internal/point_project/viz/pascal_val/';
%folder = ['/afs/cs.stanford.edu/u/abearman/www'];
base_folder = '//image-net.org/internal/point_project/viz/pascal_val/';
il_cls_con = [base_folder 'il-cls-con'];
il_cls_con_obj = [base_folder 'il-cls-con-obj'];
real1_click1_cls_con = [base_folder 'real1-click1-cls-con'];
real1_click1_cls_con_obj = [base_folder 'real1-click1-cls-con-obj'];
real1_squiggle1_cls_con_obj = [base_folder 'real1-squiggle1-cls-con-obj'];
random1_users3 = [base_folder 'random1-users3']; 
fully_supervised = [base_folder 'fully-supervised'];
gt = [base_folder 'gt'];

Nperpage = 1500;
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
	fprintf(fid, '<td>%s</td>\n<td>IMG</td>\n<td>IMG + OBJ</td>\n<td>IMG + OBJ + 1 click</td>/n<td>IMG + OBJ + 1 squiggle</td>\n<td>Fully supervised</td>\n<td>Ground truth</td>\n</tr>\n', ids{k});
        fprintf(fid,'<tr>\n<td>\n<img width=150 src=%s/%s_in.png>\n</td>\n', il_cls_con, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s_out.png>\n</td>\n', il_cls_con, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s_out.png>\n</td>\n', il_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s_out.png>\n</td>\n', real1_click1_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s_out.png>\n</td>\n', real1_squiggle1_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s_out.png>\n</td>\n', fully_supervised, ids{k});
	fprintf(fid,'<td>\n<img width=150 src=%s/%s.png>\n</td>\n</tr>\n</table>', gt, ids{k});
    end
    fclose(fid);
end
