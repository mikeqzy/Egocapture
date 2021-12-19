import numpy as np
import glob, os, sys
import pdb
import csv

import pandas as pd
import json
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# kps_labels=['Nose',
#             'Neck',
#             'Right Shoulder',
#             'Right Elbow',
#             'Right Wrist',
#             'Left Shoulder',
#             'Left Elbow',
#             'Left Wrist',
#             'Middle Hip',
#             'Right Hip',
#             'Right Knee',
#             'Right Ankle',
#             'Left Hip',
#             'Left Knee',
#             'Left Ankle',
#             'Right Eye',
#             'Left Eye',
#             'Right Ear',
#             'Left Ear',
#             'Left BigToe',
#             'Left SmallToe',
#             'Left Heel',
#             'Right BigToe',
#             'Right SmallToe',
#             'Right Heel'
#             ]

kps_labels = ['Nose',
              'Neck Bottom',
              'Right Shoulder',
              'Right Elbow',
              'Right Wrist',
              'Left Shoulder',
              'Left Elbow',
              'Left Wrist',
              'Pelvis',
              'Right Hip',
              'Right Knee',
              'Right Ankle',
              'Right Toebase',
              'Left Hip',
              'Left Knee',
              'Left Ankle',
              'Left Toebase']


resultsfile = 'data/kps_annotation.pkl'
with open(resultsfile, 'rb') as f:
    annotations = pickle.load(f)

data_root = '/local/home/zhqian/sp/data/amt_annotation_data'

for anno in tqdm(annotations):
    # pdb.set_trace()
    imgpath = os.path.join(data_root, anno['frame_name'])
    img = plt.imread(imgpath)
    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    for label in kps_labels:
        loc = anno['annotation'][label]['location']
        mean = anno['annotation'][label]['mean']

        if mean is not None:
            circ = Circle((mean[0],mean[1]),10)
            ax.add_patch(circ)

    # plt.show()
    plt.savefig(os.path.join(data_root, 'vis', anno['frame_name'].split('/')[-1]))
    plt.close()













