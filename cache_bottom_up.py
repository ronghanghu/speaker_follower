import tqdm
from env import BottomUpImageFeatures, MeanPooledImageFeatures
import pickle

detections = 20

mean_pooled_featuers = MeanPooledImageFeatures(["imagenet"])
bottom_up_features = BottomUpImageFeatures(detections)

keys = list(mean_pooled_featuers.features.keys())
missing_keys = set()
by_key = {}

for key in tqdm.tqdm(keys):
    scene, viewpoint = key.split('_')
    try:
        by_key[(scene, viewpoint)] = [nt._asdict() for nt in bottom_up_features._get_viewpoint_features(scene, viewpoint)]
    except Exception as e:
        print(e)
        print(scene, viewpoint)
        missing_keys.add((scene, viewpoint))

with open('img_features/bottom_up_10_100_d={}.pkl'.format(detections), 'wb') as f:
    pickle.dump(by_key, f)
