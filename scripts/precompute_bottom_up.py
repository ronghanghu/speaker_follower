#!/usr/bin/env python

''' Script to precompute image features using a Bottom-Up attention, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT
    and VFOV parameters. '''

import numpy as np
import cv2
import json
import math
import sys
import os
import os.path
import matplotlib.pyplot as plt
import pickle
import argparse
from multiprocessing import Process
import pprint
import time
import glob

# Caffe and MatterSim need to be on the Python path
sys.path.insert(0, 'build')
import MatterSim

bottom_up_root = '/home/anja/projects/bottom-up-attention' # TODO : change to your location
sys.path.insert(0, bottom_up_root + '/caffe/python')
sys.path.insert(0, bottom_up_root + '/tools')

import caffe

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

data_path = bottom_up_root + '/data/genome/1600-400-20'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())


cfg_from_file(bottom_up_root + '/experiments/cfgs/faster_rcnn_end2end_resnet.yml')
print('Using config:')
pprint.pprint(cfg)
assert cfg.TEST.HAS_RPN

from timer import Timer

VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10 # 36
MAX_BOXES = 100 # 36

PROTO = bottom_up_root + '/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

if MIN_BOXES == 36 and MAX_BOXES == 36:
    MODEL = bottom_up_root + '/data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel' # model for 36 features
else:
    MODEL = bottom_up_root + '/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel' # model for 10-100 features

CONV_FEATURE_DIR = 'img_features/bottom_up'
GRAPHS = 'connectivity/'

# Simulator image parameters
WIDTH=640
HEIGHT=480
VFOV=60

def get_detections_from_im(net, im, image_id, conf_thresh=0.2):
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    attr_prob = net.blobs['attr_prob'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    ############################

    #uncomment for visualizations
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(im)

    boxes = cls_boxes[keep_boxes]
    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)

    captions = [None] * len(keep_boxes)
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i]+1]
        if attr_conf[i] > attr_thresh:
             cls = attributes[attr[i]+1] + " " + cls
        captions[i] = cls
        #plt.gca().add_patch(
        #    plt.Rectangle((bbox[0], bbox[1]),
        #              bbox[2] - bbox[0],
        #              bbox[3] - bbox[1], fill=False,
        #              edgecolor='red', linewidth=2, alpha=0.5)
        #        )
        #plt.gca().text(bbox[0], bbox[1] - 2,
        #        '%s' % (cls),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=10, color='white')
    #plt.show()
    #plt.close()

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes],
        'cls_prob': np.max(cls_prob[keep_boxes][:,1:], axis=1),
        'captions': captions
    }

def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS+scan+'_connectivity.json')  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print 'Loaded %d viewpoints' % len(viewpointIds)
    return viewpointIds


def transform_img(im):
    ''' Prep opencv 3 channel image for the network '''
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob


def build_tsv(gpu_id, ids):
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.init()

    # Set up Caffe resnet
    caffe.set_device(gpu_id)#GPU_ID)
    caffe.set_mode_gpu()
    net = caffe.Net(PROTO, MODEL, caffe.TEST)    

    count = 0
    t_render = Timer()
    t_net = Timer()

    # Loop all the viewpoints in the simulator
    viewpointIds = load_viewpointids()
    it = viewpointIds
    try:
        import tqdm
        it = tqdm.tqdm(it)
    except:
        pass

    for scanId in set(scanId for scanId, _ in viewpointIds):
        scan_path = os.path.join(CONV_FEATURE_DIR, scanId)
        if not os.path.exists(scan_path):
            os.makedirs(scan_path)                
    for scanId,viewpointId in it:

        if scanId not in ids: continue

        if os.path.exists(os.path.join(CONV_FEATURE_DIR, scanId, "%s.p" % viewpointId)): continue
        print('working on: %s-%s', scanId, viewpointId)

        t_render.tic()            

        # Loop all discretized views from this location
        blobs = []
        outputs_all = [None] * VIEWPOINT_SIZE
        try:            
            for ix in range(VIEWPOINT_SIZE):
       	        if ix == 0:
	            sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
	        elif ix % 12 == 0:
	            sim.makeAction(0, 1.0, 1.0)
	        else:
	            sim.makeAction(0, 1.0, 0)
                state = sim.getState()
                assert state.viewIndex == ix
                blobs.append(state.rgb)
        except:
            print('dropping: %s-%s', scanId, viewpointId)
            continue

        t_render.toc()
        t_net.tic()
        # Run as many forward passes as necessary
        forward_passes = VIEWPOINT_SIZE
        ix = 0
        for f in range(forward_passes):             
            # Forward pass
            output = get_detections_from_im(net, blobs[ix], ix)
            ix += 1
            outputs_all[f] = output

        pickle.dump(outputs_all, open(os.path.join(CONV_FEATURE_DIR, scanId, "%s.p" % viewpointId), "wb"))
        count += 1
        t_net.toc()
        if count % 100 == 0:
            print 'Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours' %\
              (count,len(viewpointIds), t_render.average_time, t_net.average_time,
              (t_render.average_time+t_net.average_time)*len(viewpointIds)/3600)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":    

    args = parse_args()

    print('Called with args:')
    print(args)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    # Split IDs between gpus
    try:
      of = open("./list.txt", 'r')
    except:
      os.system('ls ./data/v1/scans/ > list.txt')
      of = open("./list.txt", 'r')
    ids = of.read().split('\n')
    if ids[-1] == '':
        ids = ids[0:-1]
    
    ids = [ids[i::len(gpus)] for i in range(len(gpus))]    
    procs = []

    #import ipdb; ipdb.set_trace()
    
    for i,gpu_id in enumerate(gpus):
        p = Process(target=build_tsv,
                    args=(gpu_id, ids[i]))
        p.daemon = True
        p.start()
        procs.append(p)
        #build_tsv(gpu_id, ids[i], OUTFILE_i)
    for p in procs:
        p.join()        


