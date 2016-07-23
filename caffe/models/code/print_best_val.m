function print_val_page()

imgset = '2012v';

%idspath = '/imagenetdb3/olga/data/segm_lmdb/pascal_%s.txt';
%ids = textread(sprintf(idspath,imgset),'%s');
ids = ['2007_000676'; '2007_000783'; '2007_000925'; '2007_000999'; '2007_001288'; '2007_001289'; '2007_001763'; '2007_002266'; '2007_002284'; '2007_002400'; '2007_003022'; '2007_003051'; '2007_003143'; '2007_003742'; '2007_004052'; '2007_004143'; '2007_004380'; '2007_004483'; '2007_004856'; '2007_005107'; '2007_005114'; '2007_005173'; '2007_005281'; '2007_005331'; '2007_005354'; '2007_005600'; '2007_005608'; '2007_005759'; '2007_005828'; '2007_005845'; '2007_005857'; '2007_006364'; '2007_006449'; '2007_006946'; '2007_007084'; '2007_007235'; '2007_007470'; '2007_009346'; '2007_009911'; '2008_000215'; '2008_000391'; '2008_000510'; '2008_000666'; '2008_001135'; '2008_001231';];
ids = cellstr(ids);

folder = '/imagenetdb/www/home/internal/point_project/viz/pascal_best_val';
%folder = '/imagenetdb/www/home/internal/point_project/viz/train'        
%folder = ['/afs/cs.stanford.edu/u/abearman/www'];
base_folder = '//image-net.org/internal/point_project/viz/pascal_val/';
il_cls_con = [base_folder 'il-cls-con'];
il_cls_con_obj = [base_folder 'il-cls-con-obj'];
real1_click1_cls_con = [base_folder 'real1-click1-cls-con'];
real1_click1_cls_con_obj = [base_folder 'real1-click1-cls-con-obj'];
random1_users3 = [base_folder 'random1-users3']; 
fully_supervised = [base_folder 'fully-supervised'];
gt = [base_folder 'gt'];

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
	fprintf(fid, '<td>%s</td>\n<td>IMG</td>\n<td>IMG + OBJ</td>\n<td>IMG + 1 click</td>/n<td>IMG + OBJ + 1 click</td>\n<td>Fully supervised</td>\n<td>Ground truth</td>\n</tr>\n', ids{k});
        fprintf(fid,'<tr>\n<td>\n<img width=200 src=%s/%s_in.png>\n</td>\n', il_cls_con, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', il_cls_con, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', il_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', real1_click1_cls_con, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', real1_click1_cls_con_obj, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s_out.png>\n</td>\n', fully_supervised, ids{k});
	fprintf(fid,'<td>\n<img width=200 src=%s/%s.png>\n</td>\n</tr>\n</table>', gt, ids{k});
    end
    fclose(fid);
end
