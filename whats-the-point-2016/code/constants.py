#COLOR_SCHEME_NAMES			 = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'magenta', 'cyan',
#																 'darkblue', 'darkgreen', 'goldenrod', 'maroon', 'palevioletred', 'salmon',
#																 'tomato', 'orchid', 'greenyellow', 'lavender', 'mistyrose', 'darkturquoise']

COLOR_SCHEME_HEX = ['#000000', '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF', '#A020F0', '#FF00FF', '#00FFFF',
'#00008B', '#006400', '#DAA520', '#B03060', '#DB7093', '#FA8072',
'#FF6347', '#DA70D6', '#ADFF2F', '#E6E6FA', '#FFE4B5', '#00CED1']

#COLOR_SCHEME_HEX = [
#	'#000000',
#	'#FF0000', 
#	'#FFA500',
#	'#FFFF00',
#	'#7CFC00',
#	'#000080',
#	'#663399', 
#	'#00FF7F',
#	'#00FFFF',
#	'#DB7093',
#	'#800000',
#	'#E6E6FA',
#	'#FF69B4',
#	'#FF6347',
#	'#FFB6C1',
#	'#00CED1',
#	'#87CEFA',
#	'#228B22',
#	'#FF00FF',
#	'#FFFACD',
#	'#1E90FF']	

PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))
