import numpy as np
import cv2
from os.path import join
from glob import glob
from tqdm import tqdm

from IPython import embed

data_root = '/local/home/zhqian/sp/data/calibration'

def generate_video(rec):
    rec_dir = join(data_root, rec)
    before_dir = join(rec_dir, 'fpv_render_imgs_new')
    after_dir = join(rec_dir, 'fpv_render_imgs_opt')
    before_frames = sorted(glob(join(before_dir, '*')))
    after_frames = sorted(glob(join(after_dir, '*')))
    assert len(before_frames) == len(after_frames)

    frame = cv2.imread(before_frames[0])
    h, w, _ = frame.shape
    h *= 2
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(join(rec_dir, 'holo_diff.avi'), fourcc, 30., (w, h))

    for before, after in tqdm(zip(before_frames, after_frames), total=len(before_frames)):
        before_img = cv2.imread(before)
        after_img = cv2.imread(after)
        concat_img = cv2.vconcat([before_img, after_img])
        video.write(concat_img)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    recs = ['recording_20210910_s2_01_ines_moh',
            'recording_20210910_s2_05_moh_ines',
            'recording_20210929_s2_01_ines_sara',
            'recording_20210929_s2_02_sara_ines',
            'recording_20210929_s2_05_ines_sara']
    for rec in recs:
        generate_video(rec)