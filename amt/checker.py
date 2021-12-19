import pandas as pd
import numpy as np
import json
from IPython import embed

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

def calc_dist(loc):
    dist = loc - np.mean(loc, axis=0)[None, :]
    dist = np.sqrt(np.sum(dist ** 2, axis=1))
    return dist

def checker(imgname=None):
    df = pd.read_csv('results/amt_201-1200_batch_results.csv')
    full_names = list(df['Input.image_url'].unique())
    imgnames = [x.split('/')[-1] for x in full_names]
    full_name = full_names[imgnames.index(imgname)]
    sframe = df.loc[df['Input.image_url'] == full_name]['Answer.annotatedResult.keypoints']
    kps = {}
    for label in kps_labels:
        kps[label] = []
    for allkps in sframe:
        allkps = json.loads(allkps)
        for anno in allkps:
            label = anno['label']
            loc = (anno['x'], anno['y'])
            kps[label].append(loc)
    embed()

if __name__ == '__main__':
    checker('132764507374858308_frame_02873.jpg')
