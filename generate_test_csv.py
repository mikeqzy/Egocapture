import numpy as np
import pandas as pd
from glob import glob
import os
from os.path import join
from IPython import embed

data_root = '/local/home/zhqian/sp/data/amt_annotation_data'
web_root = 'https://egocap.s3.us-west-2.amazonaws.com/amt_annotation_data/'

if __name__ == '__main__':
    recordings = glob(join(data_root, 'recording*'))
    image_url = []
    target_url = []
    for rec in recordings:
        image_dir = join(rec, 'images')
        target_path = join(rec, 'target.jpg')
        images = glob(join(image_dir, '*'))
        images = [join(web_root, os.path.relpath(x, data_root)) for x in images]
        target_path = join(web_root, os.path.relpath(target_path, data_root))
        targets = [target_path] * len(images)
        image_url.extend(images)
        target_url.extend(targets)
    df = pd.DataFrame(list(zip(image_url, target_url)),
                      columns=['image_url', 'target_url'])
    df.to_csv(join(data_root, 'amt_annotation.csv'), index=False)