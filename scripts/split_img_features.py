import glob
import os.path
import numpy as np
import re

base_path = '/data2/matterport3d/skybox_image_features/imagenet_convolutional/'

if __name__ == "__main__":
    for scene_dir in glob.glob(os.path.join(base_path, '*')):
        print("scene_dir: {}".format(scene_dir))
        for feat_path in glob.glob(os.path.join(scene_dir, '*.npy')):
            fname, ext = os.path.splitext(os.path.basename(feat_path))
            assert ext == ".npy"
            if re.match(r"^[a-f0-9]+$", fname):
                feats = np.load(feat_path)
                for feat_ix, split_feat in enumerate(feats):
                    out_fname = os.path.join(scene_dir, "{}_{}.npy".format(fname, feat_ix))
                    np.save(out_fname, split_feat)
            else:
                print("bad file: {}".format(feat_path))
