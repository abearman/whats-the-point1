import json


RESULTS_FILE = "/afs/cs/u/abearman/whats_the_point_2015/AMT_UI/results/pascal_2012tv_SBDtv_n30_acc80_t628.txt"


print "Reading results file ..."
output_json_data = open(RESULTS_FILE).read()
print "Done reading results file"
output_data = json.loads(output_json_data)
print "Done loading results json"

print "There are " + str(len(output_data)) + " HITs to parse."

f = open('/imagenetdb3/abearman/caffe/models/code/PASCAL_AMT_TIMES.txt', 'w')

counter = 1
for hit in output_data:
	print counter
	counter += 1	
	img_ids = [str(img["img_id"]) for img in hit["output"]["input"]["questions"]]
	img_outputs = hit["output"]["output"]

	for i in xrange(len(img_outputs)):
		img_id = img_ids[i]
		click_data = img_outputs[i]
		time = click_data["time"] / 1000.0  # In ms
		f.write(img_id + " " + str(time) + "\n")

f.close()		
