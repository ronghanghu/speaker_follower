''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import json
import random
import networkx as nx
import functools
import os.path
import time
import paths
import pickle
import os
import os.path

from collections import namedtuple, defaultdict

from utils import load_datasets, load_nav_graphs, structured_map, vocab_pad_idx, decode_base64, k_best_indices, try_cuda, spatial_feature_from_bbox

import torch
from torch.autograd import Variable

csv.field_size_limit(sys.maxsize)

FOLLOWER_MODEL_ACTIONS = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']

LEFT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("left")
RIGHT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("right")
UP_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("up")
DOWN_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("down")
FORWARD_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("forward")
END_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<end>")
START_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<start>")
IGNORE_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<ignore>")


FOLLOWER_ENV_ACTIONS = [
    (0,-1, 0), # left
    (0, 1, 0), # right
    (0, 0, 1), # up
    (0, 0,-1), # down
    (1, 0, 0), # forward
    (0, 0, 0), # <end>
    (0, 0, 0), # <start>
    (0, 0, 0)  # <ignore>
]

assert len(FOLLOWER_MODEL_ACTIONS) == len(FOLLOWER_ENV_ACTIONS)

WorldState = namedtuple("WorldState", ["scanId", "viewpointId", "heading", "elevation"])

BottomUpViewpoint = namedtuple("BottomUpViewpoint", ["cls_prob", "image_features", "attribute_indices", "object_indices", "spatial_features", "no_object_mask"])

def load_world_state(sim, world_state):
    sim.newEpisode(*world_state)

def get_world_state(sim):
    state = sim.getState()
    return WorldState(scanId=state.scanId,
                      viewpointId=state.location.viewpointId,
                      heading=state.heading,
                      elevation=state.elevation)

def make_sim(image_w, image_h, vfov):
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.init()
    return sim

# def encode_action_sequence(action_tuples):
#     encoded = []
#     reached_end = False
#     if action_tuples[0] == (0, 0, 0):
#         # this method can't handle a <start> symbol
#         assert all(t == (0, 0, 0) for t in action_tuples)
#     for tpl in action_tuples:
#         if tpl == (0, 0, 0):
#             if reached_end:
#                 ix = IGNORE_ACTION_INDEX
#             else:
#                 ix = END_ACTION_INDEX
#                 reached_end = True
#         else:
#             ix = FOLLOWER_ENV_ACTIONS.index(tpl)
#         encoded.append(ix)
#     return encoded

def index_action_tuple(action_tuple):
    ix, heading_chg, elevation_chg = action_tuple
    if heading_chg > 0:
        return FOLLOWER_MODEL_ACTIONS.index('right')
    elif heading_chg < 0:
        return FOLLOWER_MODEL_ACTIONS.index('left')
    elif elevation_chg > 0:
        return FOLLOWER_MODEL_ACTIONS.index('up')
    elif elevation_chg < 0:
        return FOLLOWER_MODEL_ACTIONS.index('down')
    elif ix > 0:
        return FOLLOWER_MODEL_ACTIONS.index('forward')
    else:
        return FOLLOWER_MODEL_ACTIONS.index('<end>')

class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048
    feature_dim = MEAN_POOLED_DIM

    IMAGE_W = 640
    IMAGE_H = 480
    VFOV = 60

    @staticmethod
    def from_args(args):
        feats = []
        for image_feature_type in sorted(args.image_feature_type):
            if image_feature_type == "none":
                feats.append(NoImageFeatures())
            elif image_feature_type == "bottom_up_attention":
                feats.append(BottomUpImageFeatures(
                    args.bottom_up_detections,
                    #precomputed_cache_path=paths.bottom_up_feature_cache_path,
                    precomputed_cache_dir=paths.bottom_up_feature_cache_dir,
                ))
            elif image_feature_type == "convolutional_attention":
                feats.append(ConvolutionalImageFeatures(
                    args.image_feature_datasets,
                    split_convolutional_features=True,
                    downscale_convolutional_features=args.downscale_convolutional_features
                ))
            else:
                assert image_feature_type == "mean_pooled"
                feats.append(MeanPooledImageFeatures(args.image_feature_datasets))
        return feats

    @staticmethod
    def add_args(argument_parser):
        argument_parser.add_argument("--image_feature_type", nargs="+", choices=["none", "mean_pooled", "convolutional_attention", "bottom_up_attention"], default=["mean_pooled"])
        argument_parser.add_argument("--image_attention_size", type=int)
        argument_parser.add_argument("--image_feature_datasets", nargs="+", choices=["imagenet", "places365"], default=["imagenet"], help="only applicable to mean_pooled or convolutional_attention options for --image_feature_type")
        argument_parser.add_argument("--bottom_up_detections", type=int, default=20)
        argument_parser.add_argument("--bottom_up_detection_embedding_size", type=int, default=20)
        argument_parser.add_argument("--downscale_convolutional_features", action='store_true')

    def get_name(self):
        raise NotImplementedError("get_name")

    def batch_features(self, feature_list):
        features = np.stack(feature_list)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def get_features(self, state):
        raise NotImplementedError("get_features")

class NoImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self):
        print('Image features not provided')
        self.features = np.zeros(self.feature_dim, dtype=np.float32)

    def get_features(self, state):
        return self.features

    def get_name(self):
        return "none"

class MeanPooledImageFeatures(ImageFeatures):
    def __init__(self, image_feature_datasets):
        image_feature_datasets = sorted(image_feature_datasets)
        self.image_feature_datasets = image_feature_datasets

        self.mean_pooled_feature_stores = [paths.mean_pooled_feature_store_paths[dataset]
                                           for dataset in image_feature_datasets]
        self.feature_dim = MeanPooledImageFeatures.MEAN_POOLED_DIM * len(image_feature_datasets)
        print('Loading image features from %s' % ', '.join(self.mean_pooled_feature_stores))
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        self.features = defaultdict(list)
        for mpfs in self.mean_pooled_feature_stores:
            with open(mpfs, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    assert int(item['image_h']) == ImageFeatures.IMAGE_H
                    assert int(item['image_w']) == ImageFeatures.IMAGE_W
                    assert int(item['vfov']) == ImageFeatures.VFOV
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    features = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape((ImageFeatures.NUM_VIEWS, ImageFeatures.MEAN_POOLED_DIM))
                    self.features[long_id].append(features)
        assert all(len(feats) == len(self.mean_pooled_feature_stores) for feats in self.features.values())
        self.features = {
            long_id: np.concatenate(feats, axis=1)
            for long_id, feats in self.features.items()
        }

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def get_features(self, state):
        long_id = self._make_id(state.scanId, state.location.viewpointId)
        return self.features[long_id][state.viewIndex,:]

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_mean_pooled".format(name)
        return name

class ConvolutionalImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, image_feature_datasets, split_convolutional_features=True, downscale_convolutional_features=True):
        self.image_feature_datasets = image_feature_datasets
        self.split_convolutional_features = split_convolutional_features
        self.downscale_convolutional_features = downscale_convolutional_features

        self.convolutional_feature_stores = [paths.convolutional_feature_store_paths[dataset]
                                             for dataset in image_feature_datasets]

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    @functools.lru_cache(maxsize=3000)
    def _get_convolutional_features(self, scanId, viewpointId, viewIndex):
        feats = []
        for cfs in self.convolutional_feature_stores:
            if self.split_convolutional_features:
                path = os.path.join(cfs, scanId, "{}_{}{}.npy".format(viewpointId, viewIndex, "_downscaled" if self.downscale_convolutional_features else ""))
                this_feats = np.load(path)
            else:
                # memmap for loading subfeatures
                path = os.path.join(cfs, scanId, "%s.npy" % viewpointId)
                mmapped = np.load(path, mmap_mode='r')
                this_feats = mmapped[viewIndex,:,:,:]
            feats.append(this_feats)
        if len(feats) > 1:
            return np.concatenate(feats, axis=1)
        return feats[0]

    def get_features(self, state):
        return self._get_convolutional_features(state.scanId, state.location.viewpointId, state.viewIndex)

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_convolutional_attention".format(name)
        if self.downscale_convolutional_features:
            name = name + "_downscale"
        return name

class BottomUpImageFeatures(ImageFeatures):
    PAD_ITEM = ("<pad>",)
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, number_of_detections, precomputed_cache_path=None, precomputed_cache_dir=None, image_width=640, image_height=480):
        self.number_of_detections = number_of_detections
        self.index_to_attributes, self.attribute_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_attribute_path, BottomUpImageFeatures.PAD_ITEM, add_null=True)
        self.index_to_objects, self.object_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_object_path, BottomUpImageFeatures.PAD_ITEM, add_null=False)

        self.num_attributes = len(self.index_to_attributes)
        self.num_objects = len(self.index_to_objects)

        self.attribute_pad_index = self.attribute_to_index[BottomUpImageFeatures.PAD_ITEM]
        self.object_pad_index = self.object_to_index[BottomUpImageFeatures.PAD_ITEM]

        self.image_width = image_width
        self.image_height = image_height

        self.precomputed_cache = {}
        def add_to_cache(key, viewpoints):
            assert len(viewpoints) == ImageFeatures.NUM_VIEWS
            viewpoint_feats = []
            for viewpoint in viewpoints:
                params = {}
                for param_key, param_value in viewpoint.items():
                    if param_key == 'boxes':
                        # TODO: this is for backward compatibility, remove it
                        param_key = 'spatial_features'
                        param_value = spatial_feature_from_bbox(param_value, self.image_height, self.image_width)
                    assert len(param_value) >= self.number_of_detections
                    params[param_key] = param_value[:self.number_of_detections]
                viewpoint_feats.append(BottomUpViewpoint(**params))
            self.precomputed_cache[key] = viewpoint_feats

        if precomputed_cache_dir:
            self.precomputed_cache = {}
            import glob
            for scene_dir in glob.glob(os.path.join(precomputed_cache_dir, "*")):
                scene_id = os.path.basename(scene_dir)
                pickle_file = os.path.join(scene_dir, "d={}.pkl".format(number_of_detections))
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    for (viewpoint_id, viewpoints) in data.items():
                        key = (scene_id, viewpoint_id)
                        add_to_cache(key, viewpoints)
        elif precomputed_cache_path:
            self.precomputed_cache = {}
            with open(precomputed_cache_path, 'rb') as f:
                data = pickle.load(f)
                for (key, viewpoints) in data.items():
                    add_to_cache(key, viewpoints)

    @staticmethod
    def read_visual_genome_vocab(fname, pad_name, add_null=False):
        # one-to-many mapping from indices to names (synonyms)
        index_to_items = []
        item_to_index = {}
        start_ix = 0
        items_to_add = [pad_name]
        if add_null:
            null_tp = ()
            items_to_add.append(null_tp)
        for item in items_to_add:
            index_to_items.append(item)
            item_to_index[item] = start_ix
            start_ix += 1

        with open(fname) as f:
            for index, line in enumerate(f):
                this_items = []
                for synonym in line.split(','):
                    item = tuple(synonym.split())
                    this_items.append(item)
                    item_to_index[item] = index + start_ix
                index_to_items.append(this_items)
        assert len(index_to_items) == max(item_to_index.values()) + 1
        return index_to_items, item_to_index

    def batch_features(self, feature_list):
        def transform(lst, wrap_with_var=True):
            features = np.stack(lst)
            x = torch.from_numpy(features)
            if wrap_with_var:
                x = Variable(x, requires_grad=False)
            return try_cuda(x)

        return BottomUpViewpoint(
            cls_prob=transform([f.cls_prob for f in feature_list]),
            image_features=transform([f.image_features for f in feature_list]),
            attribute_indices=transform([f.attribute_indices for f in feature_list]),
            object_indices=transform([f.object_indices for f in feature_list]),
            spatial_features=transform([f.spatial_features for f in feature_list]),
            no_object_mask=transform([f.no_object_mask for f in feature_list], wrap_with_var=False),
        )

    def parse_attribute_objects(self, tokens):
        parse_options = []
        # allow blank attribute, but not blank object
        for split_point in range(0, len(tokens)):
            attr_tokens = tuple(tokens[:split_point])
            obj_tokens = tuple(tokens[split_point:])
            if attr_tokens in self.attribute_to_index and obj_tokens in self.object_to_index:
                parse_options.append((self.attribute_to_index[attr_tokens], self.object_to_index[obj_tokens]))
        assert parse_options, "didn't find any parses for {}".format(tokens)
        # prefer longer objects, e.g. "electrical outlet" over "electrical" "outlet"
        return parse_options[0]

    @functools.lru_cache(maxsize=20000)
    def _get_viewpoint_features(self, scan_id, viewpoint_id):
        if self.precomputed_cache:
            return self.precomputed_cache[(scan_id, viewpoint_id)]

        fname = os.path.join(paths.bottom_up_feature_store_path, scan_id, "{}.p".format(viewpoint_id))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        viewpoint_features = []
        for viewpoint in data:
            top_indices = k_best_indices(viewpoint['cls_prob'], self.number_of_detections, sorted=True)[::-1]

            no_object = np.full(self.number_of_detections, True, dtype=np.uint8) # will become torch Byte tensor
            no_object[0:len(top_indices)] = False

            cls_prob = np.zeros(self.number_of_detections, dtype=np.float32)
            cls_prob[0:len(top_indices)] = viewpoint['cls_prob'][top_indices]
            assert cls_prob[0] == np.max(cls_prob)

            image_features = np.zeros((self.number_of_detections, ImageFeatures.MEAN_POOLED_DIM), dtype=np.float32)
            image_features[0:len(top_indices)] = viewpoint['features'][top_indices]

            spatial_feats = np.zeros((self.number_of_detections, 5), dtype=np.float32)
            spatial_feats[0:len(top_indices)] = spatial_feature_from_bbox(viewpoint['boxes'][top_indices], self.image_height, self.image_width)

            object_indices = np.full(self.number_of_detections, self.object_pad_index)
            attribute_indices = np.full(self.number_of_detections, self.attribute_pad_index)

            for i, ix in enumerate(top_indices):
                attribute_ix, object_ix = self.parse_attribute_objects(list(viewpoint['captions'][ix].split()))
                object_indices[i] = object_ix
                attribute_indices[i] = attribute_ix

            viewpoint_features.append(BottomUpViewpoint(cls_prob, image_features, attribute_indices, object_indices, spatial_feats, no_object))
        return viewpoint_features

    def get_features(self, state):
        viewpoint_features = self._get_viewpoint_features(state.scanId, state.location.viewpointId)
        return viewpoint_features[state.viewIndex]

    def get_name(self):
        return "bottom_up_attention_d={}".format(self.number_of_detections)

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size, beam_size):
        self.sims = []
        self.batch_size = batch_size
        self.beam_size = beam_size
        for i in range(batch_size):
            beam = []
            for j in range(beam_size):
                sim = make_sim(ImageFeatures.IMAGE_W, ImageFeatures.IMAGE_H, ImageFeatures.VFOV)
                beam.append(sim)
            self.sims.append(beam)

    def sims_view(self, beamed):
        if beamed:
            return self.sims
        else:
            return (s[0] for s in self.sims)

    def newEpisodes(self, scanIds, viewpointIds, headings, beamed=False):
        assert len(scanIds) == len(viewpointIds)
        assert len(headings) == len(viewpointIds)
        assert len(scanIds) == len(self.sims)
        world_states = []
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            world_state = WorldState(scanId, viewpointId, heading, 0)
            if beamed:
                world_states.append([world_state])
            else:
                world_states.append(world_state)
            load_world_state(self.sims[i][0], world_state)
        assert len(world_states) == len(scanIds)
        return world_states

    def getStates(self, world_states, beamed=False):
        ''' Get list of states. '''
        def f(sim, world_state):
            load_world_state(sim, world_state)
            return sim.getState()
        return structured_map(f, self.sims_view(beamed), world_states, nested=beamed)

    def makeActions(self, world_states, actions, beamed=False):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        def f(sim, world_state, action):
            index, heading, elevation = action
            load_world_state(sim, world_state)
            sim.makeAction(index, heading, elevation)
            return get_world_state(sim)
        return structured_map(f, self.sims_view(beamed), world_states, actions, nested=beamed)

    def makeSimpleActions(self, simple_indices, beamed=False):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        def f(sim, index):
            if index == 0:
                sim.makeAction(1, 0, 0)
            elif index == 1:
                sim.makeAction(0,-1, 0)
            elif index == 2:
                sim.makeAction(0, 1, 0)
            elif index == 3:
                sim.makeAction(0, 0, 1)
            elif index == 4:
                sim.makeAction(0, 0,-1)
            else:
                sys.exit("Invalid simple action %s" % index)
        structured_map(f, self.sims_view(beamed), simple_indices, nested=beamed)
        return None

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, image_features_list, batch_size=100, seed=10, splits=['train'], tokenizer=None, beam_size=1):
        self.image_features_list = image_features_list
        self.data = []
        self.scans = []
        for item in load_datasets(splits):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    self.tokenizer = tokenizer
                    new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                else:
                    self.tokenizer = None
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.set_beam_size(beam_size)
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def set_beam_size(self, beam_size):
        # warning: this will invalidate the environment, self.reset() should be called afterward!
        try:
            invalid = (beam_size != self.beam_size)
        except:
            invalid = True
        if invalid:
            self.beam_size = beam_size
            self.env = EnvBatch(self.batch_size, beam_size)

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, sort_instr_length):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        if sort_instr_length:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - state.location.point
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right

    def observe(self, world_states, beamed=False, include_teacher=True):
        #start_time = time.time()
        obs = []
        for i,states_beam in enumerate(self.env.getStates(world_states, beamed=beamed)):
            item = self.batch[i]
            obs_batch = []
            for state in states_beam if beamed else [states_beam]:
                assert item['scan'] == state.scanId
                feature = [featurizer.get_features(state) for featurizer in self.image_features_list]
                ob = {
                    'instr_id' : item['instr_id'],
                    'scan' : state.scanId,
                    'viewpoint' : state.location.viewpointId,
                    'viewIndex' : state.viewIndex,
                    'heading' : state.heading,
                    'elevation' : state.elevation,
                    'feature' : feature,
                    'step' : state.step,
                    'navigableLocations' : state.navigableLocations,
                    'instructions' : item['instructions'],
                }
                if include_teacher:
                    ob['teacher'] = self._shortest_path_action(state, item['path'][-1])
                if 'instr_encoding' in item:
                    ob['instr_encoding'] = item['instr_encoding']
                if 'instr_length' in item:
                    ob['instr_length'] = item['instr_length']
                obs_batch.append(ob)
            if beamed:
                obs.append(obs_batch)
            else:
                assert len(obs_batch) == 1
                obs.append(obs_batch[0])

        #end_time = time.time()
        #print("get obs in {} seconds".format(end_time - start_time))
        return obs

    def reset(self, sort=False, beamed=False, load_next_minibatch=True):
        ''' Load a new minibatch / episodes. '''
        if load_next_minibatch:
            self._next_minibatch(sort)
        assert len(self.batch) == self.batch_size
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        return self.env.newEpisodes(scanIds, viewpointIds, headings, beamed=beamed)

    def step(self, world_states, actions, beamed=False):
        ''' Take action (same interface as makeActions) '''
        return self.env.makeActions(world_states, actions, beamed=beamed)

    def shortest_paths_to_goals(self, starting_world_states, max_steps, expand_multistep_actions=True):
        world_states = starting_world_states
        obs = self.observe(world_states)

        all_obs = []
        all_actions = []
        for ob in obs:
            all_obs.append([ob])
            all_actions.append([])

        ended = np.array([False] * len(obs))
        for t in range(max_steps):
            actions = [ob['teacher'] for ob in obs]
            if expand_multistep_actions:
                actions = [FOLLOWER_ENV_ACTIONS[index_action_tuple(a)] for a in actions]
            world_states = self.step(world_states, actions)
            obs = self.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    all_obs[i].append(ob)
            for i,a in enumerate(actions):
                if not ended[i]:
                    all_actions[i].append(index_action_tuple(a))
                    if a == (0, 0, 0):
                        ended[i] = True
            if ended.all():
                break
        return all_obs, all_actions

    def gold_obs_actions_and_instructions(self, max_steps, load_next_minibatch=True, expand_multistep_actions=True):
        starting_world_states = self.reset(load_next_minibatch=load_next_minibatch)
        path_obs, path_actions = self.shortest_paths_to_goals(starting_world_states, max_steps, expand_multistep_actions=expand_multistep_actions)
        encoded_instructions = [obs[0]['instr_encoding'] for obs in path_obs]
        return path_obs, path_actions, encoded_instructions
