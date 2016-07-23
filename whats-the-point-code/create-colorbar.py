import numpy as np
import PIL
from PIL import Image
import skimage
import scipy
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
import matplotlib
from pylab import *
import constants


def main(argv):
	classes = (np.array(constants.PASCAL_CLASSES)).tolist()
	print classes
	colors = (np.array(constants.COLOR_SCHEME_HEX)).tolist()
	print colors

	cmap = ListedColormap(colors[::-1])
	cb = colorbar_index(ncolors=len(classes), cmap=cmap, shrink=1.0, labels=classes[::-1])
	cb.ax.tick_params(labelsize=12)
	savefig('colorbar.png')

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
	"""
	This is a convenience function to stop you making off-by-one errors
	Takes a standard colour ramp, and discretizes it,
	then draws a colour bar with correctly aligned labels
	"""
	cmap = cmap_discretize(cmap, ncolors)
	mappable = cm.ScalarMappable(cmap=cmap)
	mappable.set_array([])
	mappable.set_clim(-0.5, ncolors+0.5)
	colorbar = plt.colorbar(mappable, **kwargs)
	colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
	colorbar.set_ticklabels(range(ncolors))
	if labels:
		colorbar.set_ticklabels(labels)
	return colorbar

def cmap_discretize(cmap, N):
	"""
	Return a discrete colormap from the continuous colormap cmap.

		cmap: colormap instance, eg. cm.jet. 
		N: number of colors.

	Example
		x = resize(arange(100), (5,100))
		djet = cmap_discretize(cm.jet, 5)
		imshow(x, cmap=djet)

	"""
	if type(cmap) == str:
		cmap = get_cmap(cmap)
	colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
	colors_rgba = cmap(colors_i)
	indices = np.linspace(0, 1., N + 1)
	cdict = {}
	for ki, key in enumerate(('red', 'green', 'blue')):
		cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
	return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

if __name__ == "__main__":
	main(sys.argv)
