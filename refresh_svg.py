import os
import time
import json
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

from datetime import datetime
from torchvision import transforms, datasets
from collections import OrderedDict

for i in range(5):
	i += 1
	title = 'val_n{}'.format(i)

	with open('val_n{}.json'.format(i)) as json_file:
		progress = json.load(json_file)

	# save evaluation svg
	show_plot_l = ['percision', 'recall', 'accuracy', 'specificity', 'f1', 'loss']
	color_l = ['b', 'r', 'g', 'c', 'm', 'y']
	fig = plt.figure(figsize=(24,9))
	fig.suptitle(title, fontsize="x-large")
	for index, key in enumerate(show_plot_l):
	    plt.subplot(2, 3, index+1)
	    plt.title(key)
	    for phase in ['train', 'val']:
	        color = (color_l[index] + '-') if phase == 'val' else '--'
	        y = progress[phase][key]
	        x = range(len(y))
	        plt.gca().set_ylim(bottom=0, top=1)
	        plt.plot(x,y, color)
	svg_path = os.path.join("{}_.svg".format(title))
	plt.savefig(svg_path)
