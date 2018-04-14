from env import R2RBatch, ImageFeatures
from collections import defaultdict
import random
import json

def flatten(lol):
    return [x for lst in lol for x in lst]

def get_scans(data):
    return set(inst['scan'] for inst in data)

def get_route_id(instr_id):
    route, inst = instr_id.split('_')
    return int(route)

def group_by_route(data):
    insts_by_route = defaultdict(list)
    scans_by_route = {}
    for inst in data:
        route_id = get_route_id(inst['instr_id'])
        scan = inst['scan']
        insts_by_route[route_id].append(inst)
        if route_id in scans_by_route:
            assert scans_by_route[route_id] == scan
        else:
            scans_by_route[route_id] = scan
    assert set(scans_by_route.keys()) == set(insts_by_route.keys())
    return insts_by_route, scans_by_route

def partition(data, scan_train_percent=0.94, route_train_percent=0.94, seed=1, N_train_subset_routes=200):
    random.seed(seed)
    scans = list(sorted(get_scans(data)))
    random.shuffle(scans)

    N_train_scans = int(len(scans) * scan_train_percent)
    seen_scans = set(scans[:N_train_scans])
    held_out_scans = set(scans[N_train_scans:])
    assert len(seen_scans) + len(held_out_scans) == len(scans)

    #insts_by_route, scans_by_route = group_by_route(data)

    val_unseen_routes = []
    seen_routes = []

    for route in data:
        if route['scan'] in seen_scans:
            seen_routes.append(route)
        else:
            val_unseen_routes.append(route)

    random.shuffle(seen_routes)

    N_train_routes = int(len(seen_routes) * route_train_percent)
    train_routes = seen_routes[:N_train_routes]
    val_seen_routes = seen_routes[N_train_routes:]

    train_subset_routes = train_routes[:N_train_subset_routes]

    assert len(train_routes) + len(val_seen_routes) + len(val_unseen_routes) == len(data)
    assert len(get_scans(train_routes) & get_scans(val_unseen_routes)) == 0
    print("len(train_routes): {}".format(len(train_routes)))
    print("len(train_subset_routes): {}".format(len(train_subset_routes)))
    print("len(val_seen_routes): {}".format(len(val_seen_routes)))
    print("len(val_unseen_routes): {}".format(len(val_unseen_routes)))

    train_scans = get_scans(train_routes)
    unseen_scan_routes = [inst for inst in val_seen_routes if inst['scan'] not in train_scans]

    print("num instances in val_seen without scans in train: {}".format(len(unseen_scan_routes)))

    return train_subset_routes, train_routes, val_seen_routes, val_unseen_routes

if __name__ == "__main__":
    image_features = ImageFeatures("none", None, None)

    BASE_PATH_TEMPLATE = "tasks/R2R/data/R2R_{}.json"

    def load(split_name):
        with open(BASE_PATH_TEMPLATE.format(split_name)) as f:
            return json.load(f)

    train = load("train")
    val_seen = load("val_seen")
    val_unseen = load("val_unseen")

    train_scans = get_scans(train)
    val_seen_scans = get_scans(val_seen)
    val_unseen_scans = get_scans(val_unseen)

    sub_train_subset, sub_train, sub_val_seen, sub_val_unseen = partition(train)

    for insts, name in [
        (sub_train, 'sub_train'),
        (sub_train_subset, 'sub_train_subset'),
        (sub_val_seen, 'sub_val_seen'),
        (sub_val_unseen, 'sub_val_unseen'),
    ]:
        path = 'tasks/R2R/data/R2R_{}.json'.format(name)
        with open(path, 'w') as f:
            json.dump(insts, f, sort_keys=True, indent=4, separators=(',', ':'))
