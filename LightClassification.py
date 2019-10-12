from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import math
import time
import os
def distance(c1, c2):
    (r1,g1,b1) = c1
    (r2,g2,b2) = c2
    return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)
start=time.time()
NUM_CLUSTERS = 5
path="/home/omer/Desktop/Classification/traffic light/traffic_light_images/test/green" 
for filename in os.listdir(path):
	print('reading image')
	im = Image.open(filename)
	im = im.resize((150, 150))      # optional, to reduce time
	ar = np.asarray(im)
	shape = ar.shape
	ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
	print (ar)
	print('finding clusters')
	codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
	#print('cluster centres:\n', codes)
	"""
	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

	index_max = scipy.argmax(counts)                    # find most frequent
	peak = codes[index_max-1]
	colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
	print('most frequent is %s (#%s)' % (peak, colour))
	"""
	colors = list(codes)
	sorted_colors_green = sorted(colors, key=lambda colors: distance(colors, (0,255,0)))
	#print("	GREEN----------------------------------------------------")
	#print(sorted_colors_green)
	sorted_colors_red = sorted(colors, key=lambda colors: distance(colors, (255,0,0)))
	#print("	RED----------------------------------------------------")
	#print(sorted_colors_red)
	"""
	print("---------*---------------*---------------*-------------*------------****")
	print("distance1:",distance(sorted_colors_green[0], (0,255,0)))
	print("distance2:",distance(sorted_colors_green[1], (0,255,0)))
	print("distance3:",distance(sorted_colors_green[2], (0,255,0)))
	print("distance4:",distance(sorted_colors_green[3], (0,255,0)))
	print("distance5:",distance(sorted_colors_green[4], (0,255,0)))
	"""
	closest_color_red = sorted_colors_red[0]
	distance_red= distance(closest_color_red, (255,0,0))
	closest_color_green = sorted_colors_green[0]
	distance_green=distance(closest_color_green, (0,255,0))
	#print("closest color red:",closest_color_red,"distance:",distance_red)
	#print("closest color green:",closest_color_green,"distance:",distance_green)
	if distance_red>distance_green:
		print("Traffic Light is green")
	else:
		print("Traffic Light is red")
	end=time.time()
	#print("time:",(end-start))
